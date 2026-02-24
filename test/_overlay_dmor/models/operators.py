
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDiff(nn.Module):
    # Strong edge-difference operator (cannot collapse to simple 3x3)
    def __init__(self, in_channels):
        super().__init__()
        self.pw = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        blur = F.avg_pool2d(x, 3, stride=1, padding=1)
        return self.pw(x - blur)


class CenterDiffConv(nn.Module):
    # True center-difference (CDC-like)
    def __init__(self, in_channels):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                            groups=in_channels, bias=False)

    def forward(self, x):
        w = self.dw.weight
        sum_w = w.sum(dim=(2, 3), keepdim=True)
        center = w[:, :, 1:2, 1:2]
        diff_w = w.clone()
        diff_w[:, :, 1:2, 1:2] = center - (sum_w - center)
        return F.conv2d(x, diff_w, padding=1, groups=self.dw.groups)


class DirectionAware(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.h = nn.Conv2d(in_channels, in_channels, (1, 3),
                           padding=(0, 1), groups=in_channels, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, (3, 1),
                           padding=(1, 0), groups=in_channels, bias=False)

    def forward(self, x):
        return self.h(x) + self.v(x)


class DilatedContext(nn.Module):
    # Stronger multi-scale separation (d=2 + d=4)
    def __init__(self, in_channels):
        super().__init__()
        self.d2 = nn.Conv2d(in_channels, in_channels, 3, padding=2,
                            dilation=2, groups=in_channels, bias=False)
        self.d4 = nn.Conv2d(in_channels, in_channels, 3, padding=4,
                            dilation=4, groups=in_channels, bias=False)

    def forward(self, x):
        return self.d2(x) + self.d4(x)


class EdgePreserveSmooth(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        smooth = self.avg(x)
        return x - smooth


class PlainDW3x3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3,
                              padding=1, groups=in_channels, bias=False)

    def forward(self, x):
        return self.conv(x)


def build_operator_pool(channels, enabled_ops=None, pool_mode="dmor"):
    channels = int(channels)

    if pool_mode == "all3x3":
        ops = [PlainDW3x3(channels) for _ in range(5)]
    else:
        ops = [
            LearnableDiff(channels),
            CenterDiffConv(channels),
            DirectionAware(channels),
            DilatedContext(channels),
            EdgePreserveSmooth(channels),
        ]

    if enabled_ops is not None and len(enabled_ops) > 0:
        enabled_set = set(int(i) for i in enabled_ops)
        ops = [op for idx, op in enumerate(ops) if idx in enabled_set]

    return nn.ModuleList(ops)
