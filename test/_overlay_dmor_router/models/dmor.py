from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass

from .operators import build_operator_pool


@dataclass
class RoutingConfig:
    router_mode: str = "dmor"   # uniform | global | spatial | dmor
    topk: int = 2               # only meaningful for "dmor" mode
    temperature: float = 1.0
    eps: float = 1e-6
    use_ste: bool = True


class GlobalRouter(nn.Module):
    def __init__(self, channels: int, num_ops: int):
        super().__init__()
        hidden = max(4, channels // 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_ops, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)  # [B,N,1,1]


class SpatialRouter(nn.Module):
    def __init__(self, channels: int, num_ops: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, num_ops, 1, bias=True)

    def forward(self, x):
        return self.conv(x)  # [B,N,H,W]


class DMOR(nn.Module):
    def __init__(
        self,
        channels: int,
        topk: int = 2,
        router_mode: str = "dmor",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.cfg = RoutingConfig(router_mode=str(router_mode), topk=int(topk), temperature=float(temperature))

        # operator pool from repo (5 ops)
        self.ops = build_operator_pool(self.channels)
        self.num_ops = len(self.ops)
        if self.num_ops <= 0:
            raise ValueError("Operator pool is empty.")

        # routers (create only what you need)
        if self.cfg.router_mode in ("dmor", "global"):
            self.global_router = GlobalRouter(self.channels, self.num_ops)
        if self.cfg.router_mode in ("dmor", "spatial"):
            self.spatial_router = SpatialRouter(self.channels, self.num_ops)

    def _uniform(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return torch.full((b, self.num_ops, h, w), 1.0 / self.num_ops, device=x.device, dtype=x.dtype)

    def _compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.router_mode

        if mode == "uniform":
            return self._uniform(x)

        if mode == "global":
            g_logits = self.global_router(x)  # [B,N,1,1]
            g_logits = g_logits / max(self.cfg.temperature, self.cfg.eps)
            g = torch.softmax(g_logits, dim=1)  # [B,N,1,1]
            return g.expand(-1, -1, x.shape[2], x.shape[3])  # broadcast to [B,N,H,W]

        if mode == "spatial":
            s_logits = self.spatial_router(x)  # [B,N,H,W]
            s_logits = s_logits / max(self.cfg.temperature, self.cfg.eps)
            return torch.softmax(s_logits, dim=1)

        # mode == "dmor" (global + spatial)
        g_logits = self.global_router(x)      # [B,N,1,1]
        s_logits = self.spatial_router(x)     # [B,N,H,W]
        logits = (g_logits + s_logits) / max(self.cfg.temperature, self.cfg.eps)
        return torch.softmax(logits, dim=1)

    def _apply_topk(self, weights: torch.Tensor) -> torch.Tensor:
        # Only apply Top-K for "dmor" (hybrid), consistent with your definition.
        if self.cfg.router_mode != "dmor":
            return weights
        if self.cfg.topk <= 0 or self.cfg.topk >= self.num_ops:
            return weights

        _, topk_idx = torch.topk(weights, k=self.cfg.topk, dim=1)
        mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
        w_hard = weights * mask
        w_hard = w_hard / (w_hard.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps))

        if self.cfg.use_ste:
            return w_hard - w_hard.detach() + weights
        return w_hard

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        weights = self._compute_weights(x)
        weights = self._apply_topk(weights)

        out = x.new_zeros(x.shape)
        for i, op in enumerate(self.ops):
            out = out + op(x) * weights[:, i:i+1, :, :]

        if return_weights:
            return out, weights
        return out
