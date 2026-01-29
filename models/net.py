import torch
import torch.nn as nn

from .dmor import DMOR


class TinyBackbone(nn.Module):
    """极简轻量 backbone：把 3ch → C，并保留 H,W（便于先跑通）"""
    def __init__(self, channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)  # [B,C,H,W]


class EdgeHead(nn.Module):
    """edge head：C→1"""
    def __init__(self, channels=32):
        super().__init__()
        self.head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.head(x)  # logits [B,1,H,W]


class DMOREdgeNet(nn.Module):
    """
    Backbone → DMOR → Head
    完整最小闭环网络（proposal 中的 DMOR-Edge 核心链路）
    """
    def __init__(self, channels=32, topk=0, router_mode="dmor"):
        super().__init__()
        self.backbone = TinyBackbone(channels)
        self.dmor = DMOR(channels=channels, topk=topk, router_mode=router_mode)
        self.head = EdgeHead(channels)

    def forward(self, x, return_weights=False):
        feat = self.backbone(x)
        if return_weights:
            feat2, weights = self.dmor(feat, return_weights=True)
            logits = self.head(feat2)
            return logits, weights
        else:
            feat2 = self.dmor(feat)
            logits = self.head(feat2)
            return logits
