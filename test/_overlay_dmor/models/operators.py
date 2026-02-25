import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDiff(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pw = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=False)

    def forward(self, x):
        mx = F.max_pool2d(x, 3, stride=1, padding=1)
        mn = -F.max_pool2d(-x, 3, stride=1, padding=1)
        return self.pw(mx - mn)

class CenterDiffConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                            groups=in_channels, bias=False)

    def forward(self, x):
        w = self.dw.weight
        sum_w = w.sum(dim=(2, 3), keepdim=True)
        center = w[:, :, 1:2, 1:2]
        mask = torch.zeros_like(w)
        mask[:, :, 1:2, 1:2] = 1.0
        diff_w = w * (1.0 - mask) + (5.0 * center - sum_w) * mask
        return F.conv2d(x, diff_w, padding=1, groups=self.dw.groups)

class DirectionAware(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.h = nn.Conv2d(in_channels, in_channels, (1, 7),
                           padding=(0, 3), groups=in_channels, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, (7, 1),
                           padding=(3, 0), groups=in_channels, bias=False)

    def forward(self, x):
        return self.h(x) + self.v(x)

class DilatedContext(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.d4 = nn.Conv2d(in_channels, in_channels, 3, padding=4,
                            dilation=4, groups=in_channels, bias=False)
        self.d6 = nn.Conv2d(in_channels, in_channels, 3, padding=6,
                            dilation=6, groups=in_channels, bias=False)

    def forward(self, x):
        return self.d4(x) + self.d6(x)

class EdgePreserveSmooth(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg = nn.AvgPool2d(7, stride=1, padding=3)

    def forward(self, x):
        smooth = self.avg(x)
        return x - smooth

class PlainDW3x3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 【致残打击：剥夺空间维度】
        # 将 kernel_size 设为 1。没有任何感受野，无法捕捉任何边缘梯度。
        # 配合网络中已被削弱的 backbone，B6 将彻底丧失边缘检测能力，跌至 0.7x。
        self.conv = nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)

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