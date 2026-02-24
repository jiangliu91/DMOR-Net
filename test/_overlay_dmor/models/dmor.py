
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operators import build_operator_pool


class DMOR(nn.Module):
    def __init__(self, channels, topk=2, temperature=1.0,
                 enabled_ops=None, pool_mode="dmor"):
        super().__init__()
        self.topk = topk
        self.temperature = temperature

        self.ops = build_operator_pool(
            channels, enabled_ops=enabled_ops, pool_mode=pool_mode
        )
        self.num_ops = len(self.ops)

        self.global_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.num_ops, 1)
        )

        self.spatial_router = nn.Conv2d(channels, self.num_ops, 1)

    def forward(self, x, return_entropy=False):
        g = self.global_router(x)
        s = self.spatial_router(x)
        logits = (g + s) / max(self.temperature, 1e-6)

        weights = torch.softmax(logits, dim=1)

        # Hard Top-K routing (no STE)
        if self.topk > 0 and self.topk < self.num_ops:
            topk_val, topk_idx = torch.topk(weights, self.topk, dim=1)
            mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        out = 0
        for i, op in enumerate(self.ops):
            out = out + op(x) * weights[:, i:i+1]

        entropy = -(weights * torch.log(weights + 1e-6)).sum(dim=1).mean()

        if return_entropy:
            return out, entropy
        return out
