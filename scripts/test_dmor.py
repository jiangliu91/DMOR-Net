# scripts/test_dmor.py
import os
import sys

# 把项目根目录加入 Python 搜索路径
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import torch
from models.dmor import DMOR


def balanced_bce(pred, gt):
    # pred is probability in [0,1]
    pos = gt.sum().clamp_min(1.0)
    neg = (1 - gt).sum().clamp_min(1.0)
    beta = neg / (pos + neg)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = -beta * (1 - gt) * torch.log(1 - pred) - (1 - beta) * gt * torch.log(pred)
    return loss.mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ===== Dummy feature input =====
    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)
    gt = (torch.rand(B, 1, H, W, device=device) > 0.9).float()

    # ===== DMOR =====
    dmor = DMOR(channels=C, topk=0, router_mode="dmor").to(device)
    dmor.train()

    # ===== Forward =====
    feat, routing = dmor(x, return_weights=True)

    # ===== Assertions: shape checks =====
    assert feat.shape == (B, C, H, W), f"feat shape mismatch: {feat.shape}"
    assert routing.ndim == 4, f"routing should be 4D [B,N,H,W], got {routing.ndim}D"
    assert routing.shape[0] == B and routing.shape[2] == H and routing.shape[3] == W, f"routing shape mismatch: {routing.shape}"

    # ===== Temporary edge head (no extra params) =====
    pred = torch.sigmoid(feat.mean(dim=1, keepdim=True))  # [B,1,H,W]

    # ===== Loss =====
    loss = balanced_bce(pred, gt)
    assert torch.isfinite(loss).all(), "loss is not finite"

    # ===== Backward =====
    opt = torch.optim.Adam(dmor.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    # Check gradients exist and finite
    grads_ok = False
    for p in dmor.parameters():
        if p.grad is not None:
            if torch.isfinite(p.grad).all():
                grads_ok = True
                break
    assert grads_ok, "No finite gradients found in DMOR parameters"

    opt.step()
    print("✅ DMOR training sanity passed")


if __name__ == "__main__":
    main()
