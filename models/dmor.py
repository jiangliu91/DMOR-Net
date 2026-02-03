# models/dmor.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .operators import build_operator_pool


@dataclass
class RoutingConfig:
    router_mode: str = "dmor"     # "dmor" or "uniform"
    topk: int = 0                # 0 = dense, K>0 = sparse top-k
    temperature: float = 1.0     # softmax temperature
    eps: float = 1e-6            # numerical safety


class GlobalRouter(nn.Module):
    """
    Global router: outputs [B,N] as a global prior for operator preference.
    Implemented via GAP + 1x1 "MLP": C -> hidden -> N
    """
    def __init__(self, channels: int, num_ops: int, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_ops, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.pool(x)               # [B,C,1,1]
        logits = self.mlp(g)           # [B,N,1,1]
        return logits.squeeze(-1).squeeze(-1)  # [B,N]


class SpatialRouter(nn.Module):
    """Spatial router: pixel-wise logits [B,N,H,W] via 1x1 conv."""
    def __init__(self, channels: int, num_ops: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, num_ops, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B,N,H,W]


class DMOR(nn.Module):
    """
    DMOR: Dynamic Modulated Operator Router

    Input : x [B,C,H,W]
    Output: out [B,C,H,W]
    Optional: return_weights -> weights [B,N,H,W]
    """
    def __init__(
        self,
        channels: int = 32,
        topk: int = 0,
        router_mode: str = "dmor",
        *,
        temperature: float = 1.0,
        dilation: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = int(channels)

        self.ops = build_operator_pool(self.channels, dilation=dilation)
        self.num_ops = len(self.ops)

        self.cfg = RoutingConfig(
            router_mode=str(router_mode),
            topk=int(topk),
            temperature=float(temperature),
            eps=float(eps),
        )

        if self.cfg.router_mode not in ("dmor", "uniform"):
            raise ValueError(f"router_mode must be 'dmor' or 'uniform', got {self.cfg.router_mode}")
        if self.cfg.topk < 0 or self.cfg.topk > self.num_ops:
            raise ValueError(f"topk must be in [0,{self.num_ops}], got {self.cfg.topk}")
        if self.cfg.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.cfg.temperature}")
        if self.cfg.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.cfg.eps}")

        # Routers (always instantiated to keep code simple and allow mode switching)
        self.global_router = GlobalRouter(self.channels, self.num_ops)
        self.spatial_router = SpatialRouter(self.channels, self.num_ops)

    def _compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return routing weights [B,N,H,W].
        - uniform: constant 1/N
        - dmor: softmax((global + spatial)/T)
        """
        B, C, H, W = x.shape
        N = self.num_ops

        if self.cfg.router_mode == "uniform":
            return x.new_full((B, N, H, W), 1.0 / float(N))

        g = self.global_router(x).view(B, N, 1, 1)    # [B,N,1,1]
        s = self.spatial_router(x)                    # [B,N,H,W]
        logits = (g + s) / self.cfg.temperature
        return torch.softmax(logits, dim=1)

    def _apply_topk(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply per-pixel Top-K sparsification with renormalization.
        Only active for router_mode='dmor' and 0<topk<num_ops.
        """
        if self.cfg.router_mode == "uniform":
            return weights
        if self.cfg.topk == 0 or self.cfg.topk >= self.num_ops:
            return weights

        idx = torch.topk(weights, k=self.cfg.topk, dim=1).indices  # [B,K,H,W]
        mask = torch.zeros_like(weights)
        mask.scatter_(1, idx, 1.0)

        w = weights * mask
        denom = w.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)
        return w / denom

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Memory-optimized fusion:
          out = sum_i op_i(x) * weights[:,i]
        Avoids stacking [B,N,C,H,W].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={x.shape[1]}")

        weights = self._compute_weights(x)
        weights = self._apply_topk(weights)

        out = None
        for i, op in enumerate(self.ops):
            feat_i = op(x)                      # [B,C,H,W]
            w_i = weights[:, i:i+1, :, :]       # [B,1,H,W]
            contrib = feat_i * w_i
            out = contrib if out is None else (out + contrib)

        if return_weights:
            return out, weights
        return out
