# scripts/test_dmor.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import torch
from models.dmor import DMOR


def balanced_bce(pred, gt):
    pos = gt.sum().clamp_min(1.0)
    neg = (1 - gt).sum().clamp_min(1.0)
    beta = neg / (pos + neg)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = -beta * (1 - gt) * torch.log(1 - pred) - (1 - beta) * gt * torch.log(pred)
    return loss.mean()


@torch.no_grad()
def check_topk_properties(weights: torch.Tensor, topk: int, eps: float = 1e-4):
    """
    weights: [B,N,H,W] after topk + renorm
    checks:
      1) sum over N is ~1 per pixel
      2) nonzero count per pixel is <= topk (or N if dense)
    """
    B, N, H, W = weights.shape
    s = weights.sum(dim=1)  # [B,H,W]
    max_err = (s - 1.0).abs().max().item()
    assert max_err < eps, f"TopK renorm failed: max |sum-1| = {max_err}"

    nz = (weights > 0).sum(dim=1)  # [B,H,W]
    if topk == 0 or topk >= N:
        # dense: should be N everywhere (or very close)
        nz_min, nz_max = int(nz.min().item()), int(nz.max().item())
        assert nz_min == N and nz_max == N, f"Dense expected nz=N={N}, got nz in [{nz_min},{nz_max}]"
    else:
        # sparse: nz should be <= topk everywhere
        nz_max = int(nz.max().item())
        assert nz_max <= topk, f"Sparse expected nz<=topk={topk}, got max nz={nz_max}"


def run_one(topk: int, router_mode: str = "dmor"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)
    gt = (torch.rand(B, 1, H, W, device=device) > 0.9).float()

    dmor = DMOR(channels=C, topk=topk, router_mode=router_mode).to(device)
    dmor.train()

    feat, routing = dmor(x, return_weights=True)

    # shape checks
    assert feat.shape == (B, C, H, W), f"feat shape mismatch: {feat.shape}"
    assert routing.shape[0] == B and routing.shape[2] == H and routing.shape[3] == W, f"routing shape mismatch: {routing.shape}"
    assert routing.ndim == 4, f"routing should be 4D [B,N,H,W], got {routing.ndim}D"

    # topk property checks (only meaningful for dmor routing)
    if router_mode != "uniform":
        check_topk_properties(routing.detach(), topk=topk)

    pred = torch.sigmoid(feat.mean(dim=1, keepdim=True))
    loss = balanced_bce(pred, gt)
    assert torch.isfinite(loss).all(), "loss is not finite"

    opt = torch.optim.Adam(dmor.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    grads_ok = False
    for p in dmor.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            grads_ok = True
            break
    assert grads_ok, "No finite gradients found in DMOR parameters"

    opt.step()
    print(f"✅ sanity passed | router={router_mode} topk={topk}")


def main():
    # test dense + sparse
    run_one(topk=0, router_mode="dmor")
    run_one(topk=2, router_mode="dmor")

    # optional: test uniform baseline (dense only)
    run_one(topk=0, router_mode="uniform")


if __name__ == "__main__":
    main()
