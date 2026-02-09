from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dmor import DMOR


class ConvBnRelu(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LiteBlock(nn.Module):
    """Lite residual block: DW + PW. If channels mismatch, no residual add."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.in_c = int(in_c)
        self.out_c = int(out_c)
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if (self.in_c == self.out_c) else y


class MultiScaleBackbone(nn.Module):
    """Stage1(H,W), Stage2(H/2,W/2), Stage3(H/4,W/4)"""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = int(base_channels)

        self.stage1_proj = ConvBnRelu(3, c)
        self.stage1_block = LiteBlock(c, c)

        self.stage2_down = ConvBnRelu(c, c * 2, s=2)
        self.stage2_block = LiteBlock(c * 2, c * 2)

        self.stage3_down = ConvBnRelu(c * 2, c * 4, s=2)
        self.stage3_block = LiteBlock(c * 4, c * 4)

    def forward(self, x):
        x1 = self.stage1_block(self.stage1_proj(x))
        x2 = self.stage2_block(self.stage2_down(x1))
        x3 = self.stage3_block(self.stage3_down(x2))
        return x1, x2, x3


class DMOREdgeNet(nn.Module):
    def __init__(self, channels: int = 32, topk: int = 2, router_mode: str = "dmor",
                 temperature: float = 1.0, backbone: str = "lite"):
        super().__init__()
        self.backbone = MultiScaleBackbone(channels)

        self.dmor1 = DMOR(channels, topk=topk, router_mode=router_mode, temperature=temperature)
        self.dmor2 = DMOR(channels * 2, topk=topk, router_mode=router_mode, temperature=temperature)
        self.dmor3 = DMOR(channels * 4, topk=topk, router_mode=router_mode, temperature=temperature)

        self.side1 = nn.Conv2d(channels, 1, 1)
        self.side2 = nn.Conv2d(channels * 2, 1, 1)
        self.side3 = nn.Conv2d(channels * 4, 1, 1)

        self.fuse = nn.Conv2d(3, 1, 1)

    def forward(self, x, return_weights: bool = False):
        f1, f2, f3 = self.backbone(x)
        img_h, img_w = x.shape[2], x.shape[3]  # FIX: H,W

        f1_d = self.dmor1(f1)
        f2_d = self.dmor2(f2)
        f3_d = self.dmor3(f3)

        o1 = self.side1(f1_d)

        o2 = self.side2(f2_d)
        o2 = F.interpolate(o2, size=(img_h, img_w), mode="bilinear", align_corners=False)

        o3 = self.side3(f3_d)
        o3 = F.interpolate(o3, size=(img_h, img_w), mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([o1, o2, o3], dim=1))

        if self.training:
            return [o1, o2, o3, fused]
        return fused
