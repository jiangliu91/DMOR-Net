import os
import sys
import argparse
import torch

# Make repo root importable (so "from models import DMOR" works)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models import DMOR  # noqa: E402


def topk_selection_stats(weights: torch.Tensor, k: int):
    """
    weights: [B, N, H, W]
    Return:
      - top1_ratio: [N] each operator selected as top1 ratio over all pixels
      - topk_ratio: [N] each operator appears in topk ratio over all pixels
    """
    B, N, H, W = weights.shape
    # Top-1
    top1 = torch.argmax(weights, dim=1)  # [B, H, W]
    top1_ratio = torch.stack([(top1 == i).float().mean() for i in range(N)], dim=0)

    # Top-K
    k = min(k, N)
    topk_idx = torch.topk(weights, k, dim=1).indices  # [B, K, H, W]
    topk_ratio = []
    for i in range(N):
        appear = (topk_idx == i).any(dim=1).float().mean()  # any over K
        topk_ratio.append(appear)
    topk_ratio = torch.stack(topk_ratio, dim=0)

    return top1_ratio, topk_ratio


def save_top1_map_png(top1_map: torch.Tensor, out_path: str):
    """
    Save a simple visualization of top1 operator index map.
    top1_map: [H, W] int64
    Output: grayscale PNG (values scaled to 0..255).
    """
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow not installed, skip saving top1 map. Install with: pip install pillow")
        return

    x = top1_map.detach().cpu()
    x = x - x.min()
    if x.max() > 0:
        x = x.float() / x.max()
    x = (x * 255).byte().numpy()
    Image.fromarray(x).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--save_vis", action="store_true", help="save top1 routing map to assets/")
    args = parser.parse_args()

    torch.manual_seed(0)

    x = torch.randn(args.batch, args.channels, args.H, args.W)

    dmor = DMOR(channels=args.channels, topk=args.topk)
    dmor.eval()

    with torch.no_grad():
        # requires you updated dmor.forward(return_weights=True)
        y, weights = dmor(x, return_weights=True)

    print("=== DMOR Sanity Check ===")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output stats: min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}")

    B, N, H, W = weights.shape
    print(f"Routing weights shape: {tuple(weights.shape)}  (B,N,H,W)")

    # Check normalization
    s = weights.sum(dim=1)  # [B,H,W]
    print(f"Weights sum check: min={s.min().item():.6f}, max={s.max().item():.6f}, mean={s.mean().item():.6f}")

    top1_ratio, topk_ratio = topk_selection_stats(weights, k=args.topk)

    print("\n--- Top-1 selection ratio per operator ---")
    for i, r in enumerate(top1_ratio.tolist()):
        print(f"Op{i+1}: {r*100:.2f}%")

    print(f"\n--- Top-{min(args.topk, N)} appearance ratio per operator ---")
    for i, r in enumerate(topk_ratio.tolist()):
        print(f"Op{i+1}: {r*100:.2f}%")

    if args.save_vis:
        os.makedirs(os.path.join(REPO_ROOT, "assets"), exist_ok=True)
        top1 = torch.argmax(weights, dim=1)[0]  # take first sample [H,W]
        out_path = os.path.join(REPO_ROOT, "assets", "top1_routing_map.png")
        save_top1_map_png(top1, out_path)
        print(f"\nSaved top1 routing map to: {out_path}")


if __name__ == "__main__":
    main()
