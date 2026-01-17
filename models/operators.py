import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDiff(nn.Module):
    """O1: Learnable Difference Operator"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        return x - self.conv(x)


class CenterDiffConv(nn.Module):
    """O2: Center-Difference Convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        weight = self.conv.weight
        center = weight[:, :, 1:2, 1:2]
        diff_weight = weight.clone()
        diff_weight[:, :, 1:2, 1:2] = center - weight.sum(dim=(2, 3), keepdim=True)
        return F.conv2d(x, diff_weight, padding=1)


class DirectionAware(nn.Module):
    """O3: Direction-aware (1x3 + 3x1)"""
    def __init__(self, channels):
        super().__init__()
        self.h = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.v = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))

    def forward(self, x):
        return self.h(x) + self.v(x)


class DilatedContext(nn.Module):
    """O4: Lightweight Multi-scale Context"""
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class EdgePreserveSmooth(nn.Module):
    """O5: Edge-preserving smoothing"""
    def __init__(self, channels):
        super().__init__()
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        return x - self.avg(x)
