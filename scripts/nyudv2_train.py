
# -*- coding: utf-8 -*-
"""
NYUDv2 training script (DMOR-Edge).

Output layout mirrors BSDS scripts:
  __out_dir__/ckpt/nyud_best.pth
  __out_dir__/ckpt/nyud_last.pth
  __out_dir__/train_log.txt
"""
from __future__ import annotations
import time, json
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from scripts._dataset_common import PairedEdgeDataset, guess_nyud_paths, guess_biped_paths

from models.net import DMOREdgeNet
from models.loss import balanced_bce_with_logits

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--modality", default="rgb", help="NYUD only: rgb or hha")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--router_mode", default="dmor")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tag", default="nyud")
    return p.parse_args()

def _maybe_resize(x, size:int):
    if not size or size <= 0: return x
    return torch.nn.functional.interpolate(x, size=(size,size), mode="bilinear", align_corners=False)

def main():
    args = parse_args()
    dev = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir/"ckpt").mkdir(parents=True, exist_ok=True)

    if "nyudv2" == "nyudv2":
        img_dir, gt_dir = guess_nyud_paths(args.data_root, args.split, args.modality)
    else:
        img_dir, gt_dir = guess_biped_paths(args.data_root, args.split)

    ds = PairedEdgeDataset(img_dir, gt_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=(dev.type=="cuda"))

    model = DMOREdgeNet(channels=args.channels, topk=args.topk, router_mode=args.router_mode, temperature=args.temperature).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and dev.type=="cuda"))

    best = 1e9
    log_path = out_dir/"train_log.txt"

    for ep in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for x, y, _ in dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            x = _maybe_resize(x, args.img_size)
            y = _maybe_resize(y, args.img_size)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, enabled=(args.amp and dev.type=="cuda")):
                out = model(x)
                loss = balanced_bce_with_logits(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item())
            n += 1
        avg = loss_sum / max(1,n)
        dt = time.time() - t0
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch {ep}/{args.epochs} loss {avg:.6f} time {dt:.1f}s\n")

        torch.save(model.state_dict(), out_dir/"ckpt"/f"{args.tag}_last.pth")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), out_dir/"ckpt"/f"{args.tag}_best.pth")

    (out_dir/"train_meta.json").write_text(json.dumps({"best_loss":best, "epochs":args.epochs}, indent=2), encoding="utf-8")
    print(f"[OK] Done. best_loss={best:.6f}")
    print(f"[CKPT] {out_dir/'ckpt'}")

if __name__ == "__main__":
    main()
