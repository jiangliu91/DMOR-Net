# models/net.py
import torch
import torch.nn as nn

from .dmor import DMOR


class TinyBackbone(nn.Module):
    """极简轻量 backbone：把 3ch → C，并保留 H,W（便于先跑通）"""
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


class EdgeHead(nn.Module):
    """edge head：C→1（logits）"""
    def __init__(self, channels: int = 32):
        super().__init__()
        self.head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # [B,1,H,W]


class DMOREdgeNet(nn.Module):
    """
    Backbone → DMOR → Head
    完整最小闭环网络（DMOR-Edge 核心链路）
    """
    def __init__(self, channels: int = 32, topk: int = 0, router_mode: str = "dmor", *, temperature: float = 1.0):
        super().__init__()
        self.backbone = TinyBackbone(channels)
        self.dmor = DMOR(channels=channels, topk=topk, router_mode=router_mode, temperature=temperature)
        self.head = EdgeHead(channels)

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
