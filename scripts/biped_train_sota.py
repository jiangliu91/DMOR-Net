import os
import sys
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import argparse
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet
from models.loss import HybridLoss

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BIPEDDataset(Dataset):
    """BIPED dataset loader aligned with BSDS500 preprocessing."""
    def __init__(self, root: str, is_train: bool = True):
        self.root = Path(root)
        self.is_train = is_train

        split = 'train' if is_train else 'test'
        self.img_dir = self.root / 'images' / split
        self.gt_dir = self.root / 'gt' / split

        valid_exts = ('.jpg', '.png')
        self.images = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_exts])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.gt_dir / f"{img_path.stem}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if self.is_train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                gt = np.ascontiguousarray(gt[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :, :])
                gt = np.ascontiguousarray(gt[::-1])

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        gt = (gt > 127).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(gt).unsqueeze(0)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BIPED training | Device: {device}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    train_ds = BIPEDDataset(args.data_root, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = DMOREdgeNet(channels=args.channels).to(device)
    criterion = HybridLoss(bce_weight=1.0, dice_weight=0.5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0

        for imgs, gts in train_loader:
            imgs, gts = imgs.to(device, non_blocking=True), gts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                preds = model(imgs)
                loss = criterion(preds, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += float(loss.item())

        tr_loss /= max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch [{epoch:03d}/{args.epochs}] | LR: {current_lr:.6f} | Loss: {tr_loss:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = Path(args.ckpt_dir) / f"dmor_BIPED_ep{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
