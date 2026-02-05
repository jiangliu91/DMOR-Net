# scripts/bsds_train.py
"""
BSDS500 training for DMOR-Edge (fixed-size batching).

This file contains NO Windows-style backslashes in strings to avoid the
common Python unicodeescape issue on Windows.

Usage example (PowerShell or CMD, single line):
  python -m scripts.bsds_train --bsds_root <BSDS500_ROOT> --out_root <OUT_ROOT> --epochs 1 --batch 4 --img_size 320 --backbone lite --router dmor --topk 2 --amp
"""
import argparse
import time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

from models.net import DMOREdgeNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_bce_with_logits(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pos = gt.sum().clamp_min(1.0)
    neg = (1.0 - gt).sum().clamp_min(1.0)
    pos_weight = (neg / pos).detach()
    return F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)


def find_gt_root(bsds_root: str) -> Path:
    p1 = Path(bsds_root) / "groundTruth"
    p2 = Path(bsds_root) / "ground_truth"
    if p1.is_dir():
        return p1
    if p2.is_dir():
        return p2
    raise FileNotFoundError("Cannot find groundTruth or ground_truth under BSDS root.")


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gt_soft(mat_path: str) -> np.ndarray:
    mat = loadmat(mat_path)
    gts = mat["groundTruth"].ravel()
    bmaps = []
    for gt in gts:
        try:
            b = gt["Boundaries"][0, 0]
        except Exception:
            b = gt[0, 0]["Boundaries"][0, 0]
        bmaps.append(b.astype(np.float32))
    return np.clip(np.mean(np.stack(bmaps, 0), 0), 0, 1)


def resize_pair(img: np.ndarray, gt: np.ndarray, size: int):
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    gt_r = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, gt_r


class BSDS500(Dataset):
    def __init__(self, root: str, split: str, img_size: int, hflip: bool = True):
        self.root = Path(root)
        self.split = split
        self.img_size = int(img_size)
        self.hflip = hflip

        img_dir = self.root / "images" / split
        gt_dir = find_gt_root(root) / split

        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs += list(img_dir.glob(ext))
        self.imgs = sorted(imgs)
        self.gts = [gt_dir / (p.stem + ".mat") for p in self.imgs]

        if not self.imgs:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = read_rgb(str(self.imgs[idx]))
        gt = read_gt_soft(str(self.gts[idx]))

        img, gt = resize_pair(img, gt, self.img_size)

        if self.split == "train" and self.hflip and random.random() < 0.5:
            img = img[:, ::-1].copy()
            gt = gt[:, ::-1].copy()

        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(gt.astype(np.float32))[None, ...]
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsds_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--backbone", type=str, default="lite", choices=["tiny", "lite"])
    ap.add_argument("--router", type=str, default="dmor", choices=["dmor", "uniform"])
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--num_workers", type=int, default=0)  # safest on Windows
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" 
    out_root = Path(args.out_root)
    ckpt_dir = out_root / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BSDS500(args.bsds_root, "train", img_size=args.img_size, hflip=True)
    val_ds   = BSDS500(args.bsds_root, "val",   img_size=args.img_size, hflip=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda")
    )

    model = DMOREdgeNet(
        channels=args.channels,
        topk=args.topk if args.router == "dmor" else 0,
        router_mode=args.router,
        temperature=args.temperature,
        backbone=args.backbone,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    best_val = 1e9
    print(f"[INFO] device={device} train={len(train_ds)} val={len(val_ds)} img_size={args.img_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                logits = model(x)
                loss = balanced_bce_with_logits(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.item())

        tr_loss /= max(len(train_loader), 1)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                vloss += float(balanced_bce_with_logits(model(x), y).item())
        vloss /= max(len(val_loader), 1)

        print(f"[EPOCH {epoch}] train_loss={tr_loss:.4f} val_loss={vloss:.4f} time={time.time()-t0:.1f}s")

        torch.save({"model": model.state_dict(), "epoch": epoch, "vloss": vloss, "args": vars(args)}, ckpt_dir / "dmor_last.pth")
        if vloss < best_val:
            best_val = vloss
            torch.save({"model": model.state_dict(), "epoch": epoch, "vloss": vloss, "args": vars(args)}, ckpt_dir / "dmor_best.pth")
            print("  ✓ saved best")

    print("✓ training finished")


if __name__ == "__main__":
    main()
