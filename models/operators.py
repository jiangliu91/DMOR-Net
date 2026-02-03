# models/operators.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDiff(nn.Module):
    """
    O1: Learnable difference (high-pass) via depthwise smoothing.
      y = x - DWConv3x3(x)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.conv(x)


class CenterDiffConv(nn.Module):
    """
    O2: Center-difference convolution (CDC-like).
    Builds a modified conv weight where the center term is replaced by (center - sum_neighborhood).

    Implementation is intentionally explicit to avoid in-place weight edits.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight  # [Cout,Cin,3,3]
        sum_w = w.sum(dim=(2, 3), keepdim=True)     # [Cout,Cin,1,1]
        center = w[:, :, 1:2, 1:2]                  # [Cout,Cin,1,1]

        diff_w = w.clone()
        diff_w[:, :, 1:2, 1:2] = center - sum_w

        return F.conv2d(x, diff_w, bias=None, stride=1, padding=1)


class DirectionAware(nn.Module):
    """
    O3: Direction-aware filtering using 1x3 + 3x1.
    Encourages horizontal/vertical edge sensitivity at low cost.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(x) + self.v(x)


class DilatedContext(nn.Module):
    """
    O4: Dilated 3x3 conv for lightweight context aggregation.
    Useful for structural continuity / weak edge confirmation.
    """
    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.dilation = int(dilation)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=self.dilation, dilation=self.dilation, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EdgePreserveSmooth(nn.Module):
    """
    O5: Fixed smoothing high-pass:
      y = x - AvgPool3x3(x)
    Keeps a stable, non-learned operator to preserve operator diversity.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.avg(x)


def build_operator_pool(channels: int, *, dilation: int = 2) -> nn.ModuleList:
    """
    Build the operator pool with a fixed, meaningful order.

    Order (fixed):
      0: LearnableDiff
      1: CenterDiffConv
      2: DirectionAware
      3: DilatedContext
      4: EdgePreserveSmooth

    Each op maps: [B,C,H,W] -> [B,C,H,W]
    """
    channels = int(channels)
    return nn.ModuleList([
        LearnableDiff(channels),
        CenterDiffConv(channels),
        DirectionAware(channels),
        DilatedContext(channels, dilation=dilation),
        EdgePreserveSmooth(channels),
    ])
