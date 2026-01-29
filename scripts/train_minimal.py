import argparse
import os
import random
import time

import torch
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


def balanced_bce_with_logits(logits, gt):
    pred = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos = gt.sum().clamp_min(1.0)
    neg = (1 - gt).sum().clamp_min(1.0)
    beta = neg / (pos + neg)
    loss = -beta * (1 - gt) * torch.log(1 - pred) - (1 - beta) * gt * torch.log(pred)
    return loss.mean()


@torch.no_grad()
def routing_stats(weights: torch.Tensor):
    """
    weights: [B, N, H, W] softmaxed routing weights (after topk masking if enabled)
    Return: (entropy_mean, top1_ratio_per_op)
    """
    eps = 1e-9
    # entropy over operator dimension
    p = weights.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=1)  # [B,H,W]
    ent_mean = ent.mean().item()

    # top-1 chosen operator index frequency
    top1 = weights.argmax(dim=1)  # [B,H,W]
    N = weights.shape[1]
    counts = torch.bincount(top1.flatten(), minlength=N).float()
    ratio = (counts / counts.sum().clamp_min(1.0)).cpu().tolist()
    return ent_mean, ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=0, help="0=dense, K=sparse top-k")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default="runs_minimal")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.save_dir, exist_ok=True)

    ds = DummyEdgeDataset()
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = DMOREdgeNet(channels=32, topk=args.topk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # parameter count (proposal: <1M)
    total_params = sum(p.numel() for p in model.parameters())
    dmor_params = sum(p.numel() for p in model.dmor.parameters())
    print(f"[INFO] device={device} seed={args.seed} topk={args.topk} lr={args.lr} iters={args.iters}")
    print(f"[INFO] params: total={total_params:,} | dmor={dmor_params:,}")

    # training loop
    model.train()
    t0 = time.time()
    last_ent = None
    last_ratio = None

    it = 0
    for x, y in dl:
        it += 1
        x, y = x.to(device), y.to(device)

        logits, weights = model(x, return_weights=True)
        loss = balanced_bce_with_logits(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # lightweight interpretability stats (proposal-aligned)
        ent_mean, ratio = routing_stats(weights.detach())
        last_ent, last_ratio = ent_mean, ratio

        if it % 10 == 0:
            print(f"iter {it:03d} | loss {loss.item():.4f} | ent {ent_mean:.4f}")

        if it >= args.iters:
            break

    dt = time.time() - t0

    # write a tiny log for ablation table
    log_path = os.path.join(args.save_dir, f"topk{args.topk}_seed{args.seed}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"topk={args.topk}\nseed={args.seed}\ndevice={device}\nlr={args.lr}\niters={args.iters}\n")
        f.write(f"total_params={total_params}\ndmor_params={dmor_params}\n")
        f.write(f"final_loss={loss.item():.6f}\nfinal_entropy={last_ent:.6f}\n")
        f.write("top1_ratio_per_op=" + ",".join([f"{r:.6f}" for r in last_ratio]) + "\n")
        f.write(f"time_sec={dt:.3f}\n")

    print(f"✅ finished | final loss {loss.item():.4f} | ent {last_ent:.4f} | log -> {log_path}")


if __name__ == "__main__":
    main()
