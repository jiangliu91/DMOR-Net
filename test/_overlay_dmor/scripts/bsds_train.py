import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet
from models.loss import HybridLoss


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_rgb(path: str):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_bsds_gt_from_mat(mat_path: str):
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    gt = mat['groundTruth']
    k = gt.shape[1] if gt.shape[0] == 1 else gt.shape[0]
    edge = None
    for i in range(k):
        item = gt[0, i] if gt.shape[0] == 1 else gt[i, 0]
        b = item['Boundaries'][0, 0].astype(np.float32)
        edge = b if edge is None else np.maximum(edge, b)
    return (edge > 0).astype(np.float32)


class BSDS500Dataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 320, augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.img_size = int(img_size)
        self.augment = bool(augment)

        self.img_dir = self.root / 'images' / split
        self.gt_dir = self.root / 'groundTruth' / split
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f'images dir not found: {self.img_dir}')
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f'groundTruth dir not found: {self.gt_dir}')

        exts = ('.jpg', '.png', '.jpeg', '.bmp')
        imgs = [p for p in self.img_dir.iterdir() if p.suffix.lower() in exts]
        imgs.sort()

        self.pairs = []
        for img_path in imgs:
            gt_path = self.gt_dir / f'{img_path.stem}.mat'
            if gt_path.is_file():
                self.pairs.append((img_path, gt_path))
        if len(self.pairs) == 0:
            raise RuntimeError('No (image, gt) pairs found.')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.pairs[idx]
        img = imread_rgb(str(img_path))
        if img is None:
            raise RuntimeError(f'Failed to read image: {img_path}')
        gt = load_bsds_gt_from_mat(str(gt_path))

        import cv2
        # resize to fixed square for stable batching
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                gt = np.ascontiguousarray(gt[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :, :])
                gt = np.ascontiguousarray(gt[::-1, :])

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img_t = torch.from_numpy(img)
        gt_t = torch.from_numpy(gt).unsqueeze(0)

        return img_t, gt_t


def freeze_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()


def main():
    p = argparse.ArgumentParser('BSDS500 DMOR training (fixed & self-contained)')
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--freeze_bn', action='store_true')
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--router_mode', default='dmor', choices=['dmor', 'uniform','global', 'spatial'])
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--backbone', default='lite', choices=['tiny', 'lite'])
    p.add_argument('--bce_weight', type=float, default=1.0)
    p.add_argument('--dice_weight', type=float, default=0.5)
    p.add_argument("--enabled_ops", nargs="+", type=int, default=None)
    p.add_argument("--pool_mode", default="dmor")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f'Using device: {device}')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print('Initializing datasets...')
    train_ds = BSDS500Dataset(args.data_root, 'train', img_size=args.img_size, augment=True)
    val_ds = BSDS500Dataset(args.data_root, 'val', img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f'Building model: backbone={args.backbone}, router={args.router_mode}, topk={args.topk}')
    model = DMOREdgeNet(channels=args.channels, topk=args.topk, router_mode=args.router_mode,
                        temperature=args.temperature, backbone=args.backbone, enabled_ops=args.enabled_ops,
                        pool_mode=args.pool_mode,).to(device)

    criterion = HybridLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            model.apply(freeze_bn)

        tr_loss = 0.0
        for imgs, gts in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                preds = model(imgs)
                loss = criterion(preds, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                gts = gts.to(device, non_blocking=True)
                with autocast(enabled=args.amp):
                    preds = model(imgs)
                    loss = criterion(preds, gts)
                va_loss += float(loss.item())
        va_loss /= max(1, len(val_loader))

        print(f'Epoch [{epoch}/{args.epochs}] | Train: {tr_loss:.4f} | Val: {va_loss:.4f}')

        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': va_loss,
            'args': vars(args),
        }
        torch.save(ckpt, Path(args.ckpt_dir) / 'dmor_last.pth')
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, Path(args.ckpt_dir) / 'dmor_best.pth')
            print(f'  >>> Saved dmor_best.pth (val_loss={best_val:.4f})')

    print('Training complete.')


if __name__ == '__main__':
    main()
