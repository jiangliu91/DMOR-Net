# scripts/train_minimal.py
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.net import DMOREdgeNet


class DummyEdgeDataset(Dataset):
    """Toy data for end-to-end sanity: random image + sparse edge mask."""
    def __init__(self, length=128, size=128, edge_prob=0.9):
        self.length = length
        self.size = size
        self.edge_prob = edge_prob

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(3, self.size, self.size)
        y = (torch.rand(1, self.size, self.size) > self.edge_prob).float()
        return x, y


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_bce_with_logits(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Stable balanced BCE using PyTorch's logits loss.
    pos_weight = neg/pos to upweight positives.
    """
    pos = gt.sum().clamp_min(1.0)
    neg = (1.0 - gt).sum().clamp_min(1.0)
    pos_weight = (neg / pos).detach()
    return F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)


@torch.no_grad()
def routing_stats(weights: torch.Tensor, topk: int):
    """
    weights: [B, N, H, W] routing weights (after topk masking if enabled)
    Returns dict:
      - entropy_mean
      - confidence_mean (mean max prob)
      - eff_num_ops_mean (mean exp(entropy))
      - winner_ratio_per_op (Top-1 operator histogram)
      - topk_membership_ratio_per_op (Top-K membership histogram)
      - collapse_ratio (max winner ratio; >0.8 often indicates collapse)
      - unused_ops (count of ops with membership < 1%)
    """
    eps = 1e-9
    p = weights.clamp_min(eps)

    ent = -(p * p.log()).sum(dim=1)  # [B,H,W]
    ent_mean = ent.mean().item()

    conf = p.max(dim=1).values
    conf_mean = conf.mean().item()

    eff_num = torch.exp(ent).mean().item()

    top1 = p.argmax(dim=1)  # [B,H,W]
    N = p.shape[1]
    counts = torch.bincount(top1.flatten(), minlength=N).float()
    winner_ratio = (counts / counts.sum().clamp_min(1.0)).cpu().tolist()

    effective_k = N if topk == 0 else topk
    topk_idx = torch.topk(p, k=effective_k, dim=1).indices  # [B,K,H,W]
    mem_counts = torch.bincount(topk_idx.flatten(), minlength=N).float()
    denom = mem_counts.sum().clamp_min(1.0)
    topk_ratio = (mem_counts / denom).cpu().tolist()

    collapse_ratio = float(max(winner_ratio)) if winner_ratio else 0.0
    unused_ops = int(sum(1 for r in topk_ratio if r < 0.01))  # <1% membership considered unused

    return {
        "entropy_mean": ent_mean,
        "confidence_mean": conf_mean,
        "eff_num_ops_mean": eff_num,
        "winner_ratio_per_op": winner_ratio,
        "topk_membership_ratio_per_op": topk_ratio,
        "collapse_ratio": collapse_ratio,
        "unused_ops": unused_ops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=str, default="dmor", choices=["dmor", "uniform"],
                        help="dmor=learned routing, uniform=no-router equal fusion")
    parser.add_argument("--topk", type=int, default=0, help="0=dense, K=sparse top-k (only for router=dmor)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default="runs_minimal")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision on cuda")
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature for dmor routing")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.save_dir, exist_ok=True)

    ds = DummyEdgeDataset()
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)

    model = DMOREdgeNet(channels=32, topk=args.topk, router_mode=args.router, temperature=args.temperature).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    dmor_params = sum(p.numel() for p in model.dmor.parameters())

    use_amp = bool(args.amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[INFO] device={device} seed={args.seed} router={args.router} topk={args.topk} "
          f"lr={args.lr} iters={args.iters} batch={args.batch} amp={use_amp} temp={args.temperature}")
    print(f"[INFO] params: total={total_params:,} | dmor={dmor_params:,}")

    model.train()
    t0 = time.time()

    last_loss = None
    last_stats = None

    it = 0
    for x, y in dl:
        it += 1
        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, weights = model(x, return_weights=True)
            loss = balanced_bce_with_logits(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # router=uniform should be treated as dense for stats
        stats_topk = 0 if args.router == "uniform" else args.topk
        last_stats = routing_stats(weights.detach(), topk=stats_topk)
        last_loss = loss.item()

        if it % 10 == 0:
            print(f"iter {it:03d} | loss {last_loss:.4f} | ent {last_stats['entropy_mean']:.4f} "
                  f"| conf {last_stats['confidence_mean']:.4f} | effN {last_stats['eff_num_ops_mean']:.2f}")

        if it >= args.iters:
            break

    dt = time.time() - t0

    # --- write logs ---
    stem = f"router{args.router}_topk{args.topk}_seed{args.seed}"
    txt_path = os.path.join(args.save_dir, f"{stem}.txt")
    json_path = os.path.join(args.save_dir, f"{stem}.json")

    payload = {
        "router": args.router,
        "topk": args.topk,
        "seed": args.seed,
        "device": device,
        "lr": args.lr,
        "iters": args.iters,
        "batch": args.batch,
        "amp": use_amp,
        "temperature": args.temperature,
        "total_params": int(total_params),
        "dmor_params": int(dmor_params),
        "final_loss": float(last_loss),
        "time_sec": float(dt),
        **(last_stats if last_stats is not None else {}),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in payload.items():
            f.write(f"{k}={v}\n")

    print(f"✅ finished | final loss {last_loss:.4f} | ent {payload.get('entropy_mean', 0.0):.4f}")
    print(f"[LOG] {txt_path}")
    print(f"[LOG] {json_path}")


if __name__ == "__main__":
    main()
