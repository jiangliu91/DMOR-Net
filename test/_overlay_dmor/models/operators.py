import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDiff(nn.Module):
    \"\"\"O1: x - DWConv(x)\"\"\"
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        nn.init.constant_(self.conv.weight, 1.0 / 9.0)

    def forward(self, x):
        return x - self.conv(x)


class CenterDiffConv(nn.Module):
    \"\"\"O2: CDC depthwise 3x3\"\"\"
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

    def forward(self, x):
        w = self.conv.weight  # [C,1,3,3]
        sum_w = w.sum(dim=(2, 3), keepdim=True)
        center = w[:, :, 1:2, 1:2]
        diff_w = w.clone()
        diff_w[:, :, 1:2, 1:2] = center - (sum_w - center)
        return F.conv2d(x, diff_w, None, stride=1, padding=1, dilation=1, groups=self.conv.groups)


class DirectionAware(nn.Module):
    \"\"\"O3: horizontal + vertical\"\"\"
    def __init__(self, in_channels: int):
        super().__init__()
        self.h = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels, bias=False)

    def forward(self, x):
        return self.h(x) + self.v(x)


class DilatedContext(nn.Module):
    \"\"\"O4: d2 + d3 context\"\"\"
    def __init__(self, in_channels: int):
        super().__init__()
        self.d2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels, bias=False)
        self.d3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, groups=in_channels, bias=False)

    def forward(self, x):
        return self.d2(x) + self.d3(x)


class EdgePreserveSmooth(nn.Module):
    \"\"\"O5: x - avgpool(x)\"\"\"
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        return x - self.avg(x)


class PlainDW3x3(nn.Module):
    \"\"\"Control baseline: plain 3x3 depthwise conv (no edge prior).\"\"\"
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

    def forward(self, x):
        return self.conv(x)


def build_operator_pool(channels: int, enabled_ops: list[int] | None = None, pool_mode: str = "dmor") -> nn.ModuleList:
    \"\"\"Build operator pool.

    pool_mode:
      - "dmor": designed 5 operators (O1~O5)
      - "all3x3": replace all 5 operators with PlainDW3x3 (B6)
    enabled_ops:
      - None/empty: keep all
      - list of indices (0..4): keep selected
    \"\"\"
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
        enabled_ops = [int(i) for i in enabled_ops]
        enabled_set = set(enabled_ops)
        ops = [op for idx, op in enumerate(ops) if idx in enabled_set]

    return nn.ModuleList(ops)
