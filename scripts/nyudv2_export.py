
# -*- coding: utf-8 -*-
"""
NYUDv2 export script (DMOR-Edge): produce per-image probability PNGs (0-255).

Output:
  __out_dir__/test_png/*.png

NYUDv2:
  --input rgb|hha uses --ckpt
  --input fusion uses --ckpt_rgb + --ckpt_hha and averages predictions (late fusion)
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from scripts._dataset_common import PairedEdgeDataset, guess_nyud_paths, guess_biped_paths
from models.net import DMOREdgeNet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--router_mode", default="dmor")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--input", default="rgb", help="NYUD only: rgb|hha|fusion")
    p.add_argument("--ckpt", default="")
    p.add_argument("--ckpt_rgb", default="")
    p.add_argument("--ckpt_hha", default="")
    return p.parse_args()

@torch.no_grad()
def _infer_dir(img_dir, gt_dir, args, ckpt_path, save_dir: Path):
    dev = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    ds = PairedEdgeDataset(img_dir, gt_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=(dev.type=="cuda"))

    model = DMOREdgeNet(channels=args.channels, topk=args.topk, router_mode=args.router_mode, temperature=args.temperature).to(dev).eval()
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)

    save_dir.mkdir(parents=True, exist_ok=True)

    for x, _, stem in dl:
        x = x.to(dev)
        if args.img_size and args.img_size > 0:
            x = torch.nn.functional.interpolate(x, size=(args.img_size,args.img_size), mode="bilinear", align_corners=False)
        with torch.amp.autocast(device_type=dev.type, enabled=(args.amp and dev.type=="cuda")):
            out = model(x)
            if isinstance(out, (list,tuple)):
                out = out[-1]
            prob = torch.sigmoid(out)[0,0].float().detach().cpu().numpy()
        cv2.imwrite(str(save_dir / f"{stem[0]}.png"), (np.clip(prob,0,1)*255.0).astype(np.uint8))

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    test_png = out_dir / "test_png"

    if "nyudv2" == "nyudv2":
        if args.input.lower() == "fusion":
            if not args.ckpt_rgb or not args.ckpt_hha:
                raise SystemExit("--input fusion requires --ckpt_rgb and --ckpt_hha")
            rgb_img, rgb_gt = guess_nyud_paths(args.data_root, args.split, "rgb")
            hha_img, hha_gt = guess_nyud_paths(args.data_root, args.split, "hha")
            tmp_rgb = out_dir / "_tmp_rgb"
            tmp_hha = out_dir / "_tmp_hha"
            _infer_dir(rgb_img, rgb_gt, args, args.ckpt_rgb, tmp_rgb)
            _infer_dir(hha_img, hha_gt, args, args.ckpt_hha, tmp_hha)

            test_png.mkdir(parents=True, exist_ok=True)
            for p in tmp_rgb.glob("*.png"):
                q = tmp_hha / p.name
                if not q.exists():
                    continue
                a = cv2.imread(str(p), 0).astype(np.float32)/255.0
                b = cv2.imread(str(q), 0).astype(np.float32)/255.0
                m = np.clip((a+b)/2.0, 0, 1)
                cv2.imwrite(str(test_png/p.name), (m*255.0).astype(np.uint8))
            print(f"[OK] fusion export -> {test_png}")
            return

        if not args.ckpt:
            raise SystemExit("--ckpt is required for rgb/hha export")
        img_dir, gt_dir = guess_nyud_paths(args.data_root, args.split, args.input)
        _infer_dir(img_dir, gt_dir, args, args.ckpt, test_png)
        print(f"[OK] export -> {test_png}")
        return

    # BIPED
    if not args.ckpt:
        raise SystemExit("--ckpt is required")
    img_dir, gt_dir = guess_biped_paths(args.data_root, args.split)
    _infer_dir(img_dir, gt_dir, args, args.ckpt, test_png)
    print(f"[OK] export -> {test_png}")

if __name__ == "__main__":
    main()
