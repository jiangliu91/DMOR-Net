# models/dmor.py
from __future__ import annotations

import torch
import torch.nn as nn

from .operators import build_operator_pool


class GlobalRouter(nn.Module):
    """
    Global router: aggregates spatial info -> [B, N]
    Implemented as GAP + 1x1 conv "MLP" (C -> hidden -> N).
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
        xg = self.pool(x)          # [B,C,1,1]
        logits = self.mlp(xg)      # [B,N,1,1]
        return logits.squeeze(-1).squeeze(-1)  # [B,N]


class SpatialRouter(nn.Module):
    """Spatial router: per-pixel logits [B, N, H, W] via 1x1 conv."""
    def __init__(self, channels: int, num_ops: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, num_ops, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B,N,H,W]


class DMOR(nn.Module):
    """
    DMOR: Dynamic Mixture of Operators (edge-oriented)

    Input : x [B,C,H,W]
    Output: out [B,C,H,W]
    Optional: return_weights -> also returns weights [B,N,H,W]
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
        self.topk = int(topk)
        self.router_mode = str(router_mode)
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.ops = build_operator_pool(self.channels, dilation=dilation)
        self.num_ops = len(self.ops)

        self.global_router = GlobalRouter(self.channels, self.num_ops)
        self.spatial_router = SpatialRouter(self.channels, self.num_ops)

        if self.topk < 0 or self.topk > self.num_ops:
            raise ValueError(f"topk must be in [0, {self.num_ops}], got {self.topk}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    def _compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return routing weights [B,N,H,W].
        - uniform: equal weights
        - dmor: learned routing via (global + spatial) logits
        """
        B, C, H, W = x.shape
        N = self.num_ops

        if self.router_mode == "uniform":
            # Equal fusion baseline
            w = x.new_full((B, N, H, W), 1.0 / float(N))
            return w

        # Learned routing (dmor)
        w_global = self.global_router(x).view(B, N, 1, 1)  # [B,N,1,1]
        w_spatial = self.spatial_router(x)                 # [B,N,H,W]
        logits = (w_global + w_spatial) / self.temperature
        weights = torch.softmax(logits, dim=1)             # [B,N,H,W]
        return weights

    def _apply_topk(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k sparsification on weights [B,N,H,W].
        Keeps topk per-pixel, renormalizes over N.
        """
        if self.router_mode == "uniform":
            return weights  # no topk for uniform baseline

        if self.topk == 0 or self.topk >= self.num_ops:
            return weights  # dense

        # topk indices [B,K,H,W]
        idx = torch.topk(weights, k=self.topk, dim=1).indices
        mask = torch.zeros_like(weights)
        mask.scatter_(1, idx, 1.0)

        w = weights * mask
        denom = w.sum(dim=1, keepdim=True).clamp_min(self.eps)
        w = w / denom
        return w

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Forward:
          1) compute routing weights
          2) optional top-k sparsify
          3) fuse operator outputs

        Memory-optimized fusion:
          we do NOT stack [B,N,C,H,W]; we accumulate directly.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")

        weights = self._compute_weights(x)
        weights = self._apply_topk(weights)

        # Accumulate without stacking to reduce peak memory
        out = None
        for i, op in enumerate(self.ops):
            feat_i = op(x)                         # [B,C,H,W]
            w_i = weights[:, i:i+1, :, :]          # [B,1,H,W]
            contrib = feat_i * w_i
            out = contrib if out is None else (out + contrib)

        if return_weights:
            return out, weights
        return out
