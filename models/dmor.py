import torch
import torch.nn as nn
import torch.nn.functional as F
from .operators import *


class DMOR(nn.Module):
    def __init__(self, channels, topk=2):
        super().__init__()
        self.topk = topk

        # Operator pool
        self.ops = nn.ModuleList([
            LearnableDiff(channels),
            CenterDiffConv(channels),
            DirectionAware(channels),
            DilatedContext(channels),
            EdgePreserveSmooth(channels)
        ])

        num_ops = len(self.ops)

        # Global routing
        self.global_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_ops, 1)
        )

        # Spatial routing
        self.spatial_router = nn.Conv2d(channels, num_ops, 1)

    def forward(self, x, return_weights: bool = False):
    B, C, H, W = x.shape
    op_feats = torch.stack([op(x) for op in self.ops], dim=1)  # [B, N, C, H, W]

    # Routing weights
    w_global = self.global_router(x).view(B, -1, 1, 1)
    w_spatial = self.spatial_router(x)
    weights = torch.softmax(w_global + w_spatial, dim=1)

    # Top-K sparse routing
    if self.topk < weights.shape[1]:
        topk_vals, topk_idx = torch.topk(weights, self.topk, dim=1)
        mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    out = (weights.unsqueeze(2) * op_feats).sum(dim=1)

    if return_weights:
        return out, weights
    return out
