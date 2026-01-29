import torch
import torch.nn as nn

from .operators import build_operator_pool


class GlobalRouter(nn.Module):
    """Global routing: x -> [B, N]"""
    def __init__(self, channels: int, num_ops: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_ops, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xg = self.pool(x)          # [B, C, 1, 1]
        logits = self.mlp(xg)      # [B, N, 1, 1]
        return logits.squeeze(-1).squeeze(-1)  # [B, N]


class SpatialRouter(nn.Module):
    """Spatial routing: x -> [B, N, H, W]"""
    def __init__(self, channels: int, num_ops: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, num_ops, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)        # [B, N, H, W]


class DMOR(nn.Module):
    """
    DMOR block:
      - operator pool O1-O5
      - global + spatial routers (optional)
      - softmax across operator dimension
      - optional top-k sparse routing
      - router_mode:
          * "dmor": learned routing (original)
          * "uniform": no-router baseline (equal weights)
    """
    def __init__(self, channels: int = 32, topk: int = 0, router_mode: str = "dmor"):
        super().__init__()
        self.channels = int(channels)
        self.topk = int(topk)
        self.router_mode = str(router_mode)

        # Operators
        self.ops = build_operator_pool(self.channels)
        self.num_ops = len(self.ops)

        # Routers (only used when router_mode == "dmor")
        self.global_router = GlobalRouter(self.channels, self.num_ops)
        self.spatial_router = SpatialRouter(self.channels, self.num_ops)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        x: [B, C, H, W]
        return:
          out: [B, C, H, W]
          weights(optional): [B, N, H, W]
        """
        B, C, H, W = x.shape

        # Operator responses: [B, N, C, H, W]
        op_feats = torch.stack([op(x) for op in self.ops], dim=1)

        # ===== No-Router baseline =====
        if self.router_mode == "uniform":
            # Equal weights across operators, no top-k (topk is meaningless here)
            weights = op_feats.new_full((B, self.num_ops, H, W), 1.0 / float(self.num_ops))
        else:
            # ===== Original DMOR learned routing =====
            # Routing logits
            w_global = self.global_router(x).view(B, self.num_ops, 1, 1)  # [B, N, 1, 1]
            w_spatial = self.spatial_router(x)                            # [B, N, H, W]

            # Normalized weights
            weights = torch.softmax(w_global + w_spatial, dim=1)          # [B, N, H, W]

            # Top-K sparse routing (optional)
            if self.topk > 0 and self.topk < self.num_ops:
                _, idx = torch.topk(weights, self.topk, dim=1)
                mask = torch.zeros_like(weights)
                mask.scatter_(1, idx, 1.0)
                weights = weights * mask
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # Weighted sum over operators
        out = (weights.unsqueeze(2) * op_feats).sum(dim=1)  # [B, C, H, W]

        if return_weights:
            return out, weights
        return out
