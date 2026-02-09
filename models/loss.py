"""models/loss.py

Compatibility-first loss module for DMOR-Edge.

This file intentionally supports multiple training scripts that may expect:
  - balanced_bce_with_logits(logits, gt)  (functional)
  - dice_loss_from_logits(logits, gt)     (functional)
  - HybridLoss(bce_weight=..., dice_weight=...) OR HybridLoss(dice_weight=...)

Notes:
  - The absolute scale of the combined loss depends on weights and class-balance.
  - For comparing two train scripts, keep the same loss definition and weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _balanced_bce_one(logits: torch.Tensor, gt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    gt_f = gt.float()
    pos = gt_f.sum().clamp_min(eps)
    neg = (1.0 - gt_f).sum().clamp_min(eps)
    pos_weight = (neg / pos).detach().to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, gt_f, pos_weight=pos_weight, reduction="mean")


def balanced_bce_with_logits(preds, gt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Class-balanced BCE.

    - If preds is a Tensor: returns BCE.
    - If preds is list/tuple (deep supervision): weighted sum of BCEs.
    """
    if isinstance(preds, (list, tuple)):
        weights = [0.5, 0.7, 1.0, 1.1]
        total = 0.0
        for i, p in enumerate(preds):
            w = weights[i] if i < len(weights) else 1.0
            total = total + w * _balanced_bce_one(p, gt, eps=eps)
        return total
    return _balanced_bce_one(preds, gt, eps=eps)


def dice_loss_from_logits(preds: torch.Tensor, gt: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss from logits."""
    gt_f = gt.float()
    probs = torch.sigmoid(preds)
    b = probs.shape[0]
    probs = probs.view(b, -1)
    gt_flat = gt_f.view(b, -1)

    inter = (probs * gt_flat).sum(dim=1)
    union = probs.sum(dim=1) + gt_flat.sum(dim=1)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class HybridLoss(nn.Module):
    """Balanced BCE + Dice.

    Compatible init signatures:
      - HybridLoss(bce_weight=1.0, dice_weight=0.5)
      - HybridLoss(dice_weight=0.5)  # bce_weight defaults to 1.0
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 0.5, **kwargs):
        super().__init__()
        # accept unused kwargs to avoid breaking older scripts
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, preds, gt: torch.Tensor) -> torch.Tensor:
        bce = balanced_bce_with_logits(preds, gt)
        bce = self.bce_weight * bce

        if self.dice_weight <= 0:
            return bce

        if isinstance(preds, (list, tuple)):
            dice = dice_loss_from_logits(preds[-1], gt)
        else:
            dice = dice_loss_from_logits(preds, gt)

        return bce + self.dice_weight * dice
