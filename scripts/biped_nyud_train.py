import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import argparse
import numpy as np
import cv2
from pathlib import Path

# 获取项目根目录 (DMOR-Edge) 并加入系统路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 从 models 文件夹中正确导入你的网络和损失函数
from models.net import DMOREdgeNet
from models.loss import HybridLoss

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EdgeDataset(Dataset):
    """Robust Dataset loader for both BIPED and NYUDv2"""
    def __init__(self, root: str, dataset_type: str, is_train: bool = True, img_size: tuple = (480, 640)):
        self.root = Path(root)
        self.dataset_type = dataset_type
        self.is_train = is_train
        self.img_size = img_size
        
        # Example directory structure, adjust to your local setup
        split = 'train' if is_train else 'test'
        self.img_dir = self.root / 'images' / split
        self.gt_dir = self.root / 'gt' / split
        
        self.hha_dir = None
        if self.dataset_type == 'NYUDv2':
            self.hha_dir = self.root / 'HHA' / split
            
        valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
        self.images = [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_exts]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.gt_dir / f"{img_path.stem}.png"
        
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Standard Data Augmentation
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
        
        img_t = torch.from_numpy(img)
        gt_t = torch.from_numpy(gt).unsqueeze(0)

        if self.dataset_type == 'NYUDv2':
            hha_path = self.hha_dir / f"{img_path.stem}.png"
            hha = cv2.imread(str(hha_path), cv2.IMREAD_COLOR)
            hha = cv2.resize(hha, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            if self.is_train and random.random() < 0.5: # Apply same flip as RGB
                 hha = np.ascontiguousarray(hha[:, ::-1, :])
            hha = hha.astype(np.float32) / 255.0
            hha = hha.transpose(2, 0, 1)
            hha_t = torch.from_numpy(hha)
            return img_t, hha_t, gt_t
            
        return img_t, gt_t

class DMORFusionWrapper(nn.Module):
    """Wrapper to handle RGB and HHA inputs simultaneously for NYUDv2"""
    def __init__(self, channels=32):
        super().__init__()
        self.rgb_net = DMOREdgeNet(channels=channels)
        self.hha_net = DMOREdgeNet(channels=channels)
        # 1x1 Conv to fuse the final outputs
        self.fusion_layer = nn.Conv2d(2, 1, 1)

    def forward(self, rgb, hha):
        out_rgb = self.rgb_net(rgb)
        out_hha = self.hha_net(hha)
        
        if self.training:
            # Fuse the final layer of both streams
            fused = self.fusion_layer(torch.cat([out_rgb[-1], out_hha[-1]], dim=1))
            return out_rgb, out_hha, [fused] # Returning expected structure
        else:
            fused = self.fusion_layer(torch.cat([out_rgb, out_hha], dim=1))
            return out_rgb, out_hha, fused

class EnhancedEdgeLoss(nn.Module):
    def __init__(self, bce_weight=1.0, rank_weight=0.5):
        super().__init__()
        self.hybrid_loss = HybridLoss(bce_weight=bce_weight, dice_weight=0.5)
        self.rank_weight = rank_weight

    def forward(self, preds, targets):
        total_loss = 0.0
        # Handle deep supervision lists
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
            
        for pred in preds:
            # Base Hybrid (BCE + Dice)
            base_loss = self.hybrid_loss(pred, targets)
            
            # Ranking Loss to suppress background noise
            mask = targets > 0.5
            pred_sig = torch.sigmoid(pred)
            pos_preds = pred_sig[mask]
            neg_preds = pred_sig[~mask]
            
            rank_loss = 0.0
            if len(pos_preds) > 0 and len(neg_preds) > 0:
                # Random sampling to prevent OOM on large images
                if len(pos_preds) > 2000: pos_preds = pos_preds[torch.randperm(len(pos_preds))[:2000]]
                if len(neg_preds) > 2000: neg_preds = neg_preds[torch.randperm(len(neg_preds))[:2000]]
                rank_loss = torch.mean(torch.relu(neg_preds.view(-1, 1) - pos_preds.view(1, -1) + 0.3))
                
            total_loss += (base_loss + self.rank_weight * rank_loss)
        return total_loss

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_nyud = (args.dataset == 'NYUDv2')
    
    print(f"Initializing Training for {args.dataset} on {device}")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    img_size = (480, 640) if is_nyud else (720, 1280)
    train_dataset = EdgeDataset(args.data_root, args.dataset, is_train=True, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if is_nyud:
        model = DMORFusionWrapper(channels=args.channels).to(device)
    else:
        model = DMOREdgeNet(channels=args.channels).to(device)

    # High-performance optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = EnhancedEdgeLoss()
    scaler = GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=args.amp):
                if is_nyud:
                    img_rgb, img_hha, targets = data
                    targets = targets.to(device, non_blocking=True)
                    out_rgb, out_hha, out_fusion = model(img_rgb.to(device), img_hha.to(device))
                    
                    # Calculate loss across both modalities and fused output
                    loss_rgb = criterion(out_rgb, targets)
                    loss_hha = criterion(out_hha, targets)
                    loss_fuse = criterion(out_fusion, targets)
                    loss = loss_rgb + loss_hha + loss_fuse
                else:
                    img_rgb, targets = data
                    targets = targets.to(device, non_blocking=True)
                    preds = model(img_rgb.to(device))
                    loss = criterion(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tr_loss += float(loss.item())

        tr_loss /= max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] | LR: {current_lr:.6f} | Train Loss: {tr_loss:.4f}")

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt_path = Path(args.ckpt_dir) / f"dmor_{args.dataset}_ep{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMOR High-Performance Edge Training")
    parser.add_argument("--dataset", type=str, required=True, choices=['BIPED', 'NYUDv2'])
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_freq", type=int, default=5)
    train(parser.parse_args())