import os
import sys

# 把项目根目录加入 Python 搜索路径
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import torch
from models.dmor import DMOR



def balanced_bce(pred, gt):
    pos = gt.sum().clamp_min(1.0)
    neg = (1 - gt).sum().clamp_min(1.0)
    beta = neg / (pos + neg)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = -beta * (1 - gt) * torch.log(1 - pred) \
           - (1 - beta) * gt * torch.log(pred)
    return loss.mean()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ===== Dummy feature input =====
    x = torch.randn(2, 32, 128, 128).to(device)
    gt = (torch.rand(2, 1, 128, 128) > 0.9).float().to(device)

    # ===== DMOR =====
    dmor = DMOR(
        channels=32,
        topk=0,                 # ⚠️ 先关 Top-K，保证闭环
        return_routing=True
    ).to(device)

    dmor.train()

    # ===== Forward =====
    feat, routing = dmor(x)

    print("[OK] feature:", feat.shape)
    print("[OK] routing:", routing.shape)

    # ===== 临时 edge head（不改 DMOR 本体）=====
    # DMOR 输出是 [B, C, H, W] → 压成 [B, 1, H, W]
    pred = torch.sigmoid(feat.mean(dim=1, keepdim=True))

    # ===== Loss =====
    loss = balanced_bce(pred, gt)
    print("[OK] loss:", loss.item())

    # ===== Backward =====
    opt = torch.optim.Adam(dmor.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    print("✅ DMOR training sanity passed")

if __name__ == "__main__":
    main()
