import os
import sys
import random
import torch
import torch.nn as nn
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class AdvancedEdgeLoss(nn.Module):
    """Texture-suppression focal loss for BIPED training."""
    def __init__(self, alpha=0.75, gamma=2.0, bce_weight=1.0, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def _dice_loss(self, preds, targets, smooth=1.0):
        preds = torch.sigmoid(preds)
        b = preds.shape[0]
        preds = preds.view(b, -1)
        targets = targets.view(b, -1)
        inter = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def _focal_bce(self, preds, targets):
        preds_sig = torch.sigmoid(preds).clamp(1e-6, 1.0 - 1e-6)
        pos_mask = (targets > 0.5).float()
        neg_mask = (targets <= 0.5).float()
        weight = self.alpha * pos_mask * torch.pow(1 - preds_sig, self.gamma) + \
                 (1 - self.alpha) * neg_mask * torch.pow(preds_sig, self.gamma)
        loss = -weight * (targets * torch.log(preds_sig) + (1 - targets) * torch.log(1 - preds_sig))
        return loss.mean()

    def forward(self, preds, targets):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        total_loss = 0.0
        weights = [0.5, 0.7, 1.0, 1.1]
        for i, pred in enumerate(preds):
            w = weights[i] if i < len(weights) else 1.0
            focal_loss = self._focal_bce(pred, targets)
            dice_loss = self._dice_loss(pred, targets)
            total_loss += w * (self.bce_weight * focal_loss + self.dice_weight * dice_loss)
        return total_loss

class BIPEDDataset(Dataset):
    def __init__(self, root: str, is_train: bool = True):
        self.img_dir = Path(root) / 'images' / ('train' if is_train else 'test')
        self.gt_dir = Path(root) / 'gt' / ('train' if is_train else 'test')
        self.images = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
        self.is_train = is_train

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.gt_dir / f"{img_path.stem}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if self.is_train:
            if random.random() < 0.5:
                img, gt = img[:, ::-1, :], gt[:, ::-1]
            if random.random() < 0.5:
                img, gt = img[::-1, :, :], gt[::-1, :]

        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        gt = np.ascontiguousarray(gt > 127).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(gt).unsqueeze(0)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BIPED training | Device: {device}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(BIPEDDataset(args.data_root, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    model = DMOREdgeNet(channels=args.channels).to(device)
    criterion = AdvancedEdgeLoss().to(device)

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
        print(f"Epoch [{epoch:03d}/{args.epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {tr_loss:.4f}")
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), Path(args.ckpt_dir) / f"dmor_biped_ult_ep{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
