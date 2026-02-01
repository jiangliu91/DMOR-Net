# models/operators.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDiff(nn.Module):
    """O1: Learnable Difference Operator (depthwise)"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High-pass style: x - smooth(x) where smooth is learnable depthwise conv
        return x - self.conv(x)


class CenterDiffConv(nn.Module):
    """
    O2: Center-Difference Convolution (CDC-like)
    Implements a center-difference re-parameterization of conv weights.

    Note: This is still learnable, but encourages sensitivity to local changes.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight  # [Cout,Cin,3,3]
        # sum over spatial dims
        sum_w = w.sum(dim=(2, 3), keepdim=True)          # [Cout,Cin,1,1]
        center = w[:, :, 1:2, 1:2]                       # [Cout,Cin,1,1]

        # Build CDC-style weight:
        # - Non-center weights keep as is
        # - Center weight becomes (center - sum_w)
        # We avoid in-place on w; create diff_w explicitly
        diff_w = w.clone()
        diff_w[:, :, 1:2, 1:2] = center - sum_w

        return F.conv2d(x, diff_w, bias=None, stride=1, padding=1, dilation=1, groups=1)


class DirectionAware(nn.Module):
    """O3: Direction-aware (1x3 + 3x1)"""
    def __init__(self, channels: int):
        super().__init__()
        self.h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(x) + self.v(x)


class DilatedContext(nn.Module):
    """O4: Lightweight multi-scale context via dilation"""
    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EdgePreserveSmooth(nn.Module):
    """O5: Edge-preserving smoothing (fixed avg pool high-pass)"""
    def __init__(self, channels: int):
        super().__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.avg(x)


def build_operator_pool(channels: int, *, dilation: int = 2) -> nn.ModuleList:
    """
    Build the operator pool O1-O5 in a fixed order.
    Returns: ModuleList of operators, each maps [B,C,H,W] -> [B,C,H,W]

    Order (fixed):
      0: LearnableDiff
      1: CenterDiffConv
      2: DirectionAware
      3: DilatedContext (dilation configurable, default=2)
      4: EdgePreserveSmooth
    """
    return nn.ModuleList([
        LearnableDiff(channels),
        CenterDiffConv(channels),
        DirectionAware(channels),
        DilatedContext(channels, dilation=dilation),
        EdgePreserveSmooth(channels),
    ])
