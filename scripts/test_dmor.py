# scripts/test_dmor.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from models import DMOR, DMOREdgeNet


@torch.no_grad()
def check_weights(weights: torch.Tensor, topk: int, eps: float = 1e-4):
    """
    weights: [B,N,H,W]
    - sum over N == 1
    - if topk>0: nonzero count <= topk
    """
    B, N, H, W = weights.shape
    s = weights.sum(dim=1)
    max_err = (s - 1.0).abs().max().item()
    assert max_err < eps, f"weights renorm failed: max |sum-1|={max_err}"

    nz = (weights > 0).sum(dim=1)
    if topk == 0 or topk >= N:
        nz_min, nz_max = int(nz.min().item()), int(nz.max().item())
        assert nz_min == N and nz_max == N, f"dense expected nz=N={N}, got [{nz_min},{nz_max}]"
    else:
        nz_max = int(nz.max().item())
        assert nz_max <= topk, f"sparse expected nz<=topk={topk}, got {nz_max}"


def test_dmor_block(topk: int, router_mode: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    dmor = DMOR(channels=C, topk=topk, router_mode=router_mode).to(device)
    dmor.train()

    y, w = dmor(x, return_weights=True)

    assert y.shape == (B, C, H, W), f"DMOR out shape mismatch: {y.shape}"
    assert w.ndim == 4 and w.shape[0] == B and w.shape[2] == H and w.shape[3] == W, f"weights shape mismatch: {w.shape}"

    if router_mode != "uniform":
        check_weights(w.detach(), topk=topk)

    pred = torch.sigmoid(y.mean(dim=1, keepdim=True))
    gt = (torch.rand(B, 1, H, W, device=device) > 0.9).float()
    loss = F.binary_cross_entropy(pred, gt)

    opt = torch.optim.Adam(dmor.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    grads_ok = any((p.grad is not None) and torch.isfinite(p.grad).all() for p in dmor.parameters())
    assert grads_ok, "DMOR: no finite gradients found"
    opt.step()

    print(f"✅ DMOR block ok | router={router_mode} topk={topk} | device={device}")


@torch.no_grad()
def test_end2end_net(topk: int, router_mode: str, backbone: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = DMOREdgeNet(channels=32, topk=topk, router_mode=router_mode, backbone=backbone).to(device)
    model.eval()

    x = torch.randn(2, 3, 256, 256, device=device)
    logits, weights = model(x, return_weights=True)

    assert logits.shape == (2, 1, 256, 256), f"DMOREdgeNet logits shape mismatch: {logits.shape}"
    assert weights is None or (weights.ndim == 4), "weights should be [B,N,H,W] or None"

    if weights is not None and router_mode != "uniform":
        check_weights(weights, topk=topk)

    print(f"✅ DMOREdgeNet ok | backbone={backbone} router={router_mode} topk={topk} | device={device}")


def main():
    test_dmor_block(topk=0, router_mode="dmor")
    test_dmor_block(topk=2, router_mode="dmor")
    test_dmor_block(topk=0, router_mode="uniform")

    test_end2end_net(topk=0, router_mode="dmor", backbone="lite")
    test_end2end_net(topk=2, router_mode="dmor", backbone="lite")
    test_end2end_net(topk=0, router_mode="uniform", backbone="tiny")


if __name__ == "__main__":
    main()
