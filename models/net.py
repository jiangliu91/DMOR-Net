# models/net.py
from __future__ import annotations

import torch
import torch.nn as nn

from .dmor import DMOR


class TinyBackbone(nn.Module):
    """极简轻量 backbone：3ch -> C，保留 H,W（用于 sanity / ablation）"""
    def __init__(self, channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)  # [B,C,H,W]


class LiteBackbone(nn.Module):
    """
    Slightly stronger but still lightweight backbone (still easy to keep <1M params).
    Uses depthwise-separable conv blocks with a residual path.
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            # DWConv
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # PWConv
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        return x + self.block(x)


class EdgeHead(nn.Module):
    """edge head: C -> 1 logits"""
    def __init__(self, channels: int = 32):
        super().__init__()
        self.head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # [B,1,H,W]


class DMOREdgeNet(nn.Module):
    """
    Backbone -> DMOR -> Head
    Minimal end-to-end network for edge detection with routing introspection.
    """
    def __init__(
        self,
        channels: int = 32,
        topk: int = 0,
        router_mode: str = "dmor",
        *,
        temperature: float = 1.0,
        backbone: str = "tiny",   # "tiny" (default) or "lite"
    ):
        super().__init__()
        self.channels = int(channels)

        if backbone not in ("tiny", "lite"):
            raise ValueError(f"backbone must be 'tiny' or 'lite', got {backbone}")

        self.backbone = TinyBackbone(self.channels) if backbone == "tiny" else LiteBackbone(self.channels)
        self.dmor = DMOR(
            channels=self.channels,
            topk=topk,
            router_mode=router_mode,
            temperature=temperature,
        )
        self.head = EdgeHead(self.channels)

    def forward(self, x: torch.Tensor, return_weights: bool = False, return_feats: bool = False):
        feat = self.backbone(x)

        if return_weights:
            feat2, weights = self.dmor(feat, return_weights=True)
            logits = self.head(feat2)
            if return_feats:
                return logits, weights, feat, feat2
            return logits, weights

        feat2 = self.dmor(feat)
        logits = self.head(feat2)
        if return_feats:
            return logits, feat, feat2
        return logits
