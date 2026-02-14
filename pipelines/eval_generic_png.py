
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Generic edge evaluation (pixel-level) for datasets with PNG GT (single annotation).

Outputs:
- ODS / OIS / AP using threshold sweep
- Precision/Recall at best ODS threshold
- optional Params (M), FLOPs (G), FPS if --ckpt is provided
"""
import json, time
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--router_mode", default="dmor")
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()

def _ap_from_pr(P, R):
    order = np.argsort(R)
    R = R[order]
    P = P[order]
    P_env = P.copy()
    for i in range(len(P_env)-2, -1, -1):
        P_env[i] = max(P_env[i], P_env[i+1])
    dR = np.diff(R)
    return float(np.sum(dR * P_env[1:]))

def eval_pixel(pred_dir: Path, gt_dir: Path, n_thresh: int = 99):
    preds = sorted(pred_dir.glob("*.png"))
    if not preds:
        raise FileNotFoundError(f"No pngs in {pred_dir}")
    thresholds = np.linspace(0.0, 1.0, n_thresh, dtype=np.float32)

    P_list, R_list, F_list = [], [], []
    per_img_bestF = []

    for t in thresholds:
        tp=fp=fn=0
        for p in preds:
            g = gt_dir / p.name
            if not g.exists():
                continue
            pr = cv2.imread(str(p), 0).astype(np.float32)/255.0
            gt = (cv2.imread(str(g), 0).astype(np.float32) > 127).astype(np.uint8)
            pb = (pr >= t).astype(np.uint8)
            tp += int((pb & gt).sum())
            fp += int((pb & (1-gt)).sum())
            fn += int(((1-pb) & gt).sum())
        prec = tp / max(1, tp+fp)
        rec  = tp / max(1, tp+fn)
        f = (2*prec*rec) / max(1e-12, prec+rec)
        P_list.append(prec); R_list.append(rec); F_list.append(f)

    P = np.array(P_list, dtype=np.float64)
    R = np.array(R_list, dtype=np.float64)
    F = np.array(F_list, dtype=np.float64)

    ods_idx = int(np.argmax(F))
    ods = float(F[ods_idx])
    ods_th = float(thresholds[ods_idx])
    ods_p = float(P[ods_idx])
    ods_r = float(R[ods_idx])

    for p in preds:
        g = gt_dir / p.name
        if not g.exists():
            continue
        pr = cv2.imread(str(p), 0).astype(np.float32)/255.0
        gt = (cv2.imread(str(g), 0).astype(np.float32) > 127).astype(np.uint8)
        bestf = 0.0
        for t in thresholds:
            pb = (pr >= t).astype(np.uint8)
            tp = int((pb & gt).sum())
            fp = int((pb & (1-gt)).sum())
            fn = int(((1-pb) & gt).sum())
            prec = tp / max(1, tp+fp)
            rec  = tp / max(1, tp+fn)
            f = (2*prec*rec) / max(1e-12, prec+rec)
            if f > bestf: bestf = f
        per_img_bestF.append(bestf)
    ois = float(np.mean(per_img_bestF)) if per_img_bestF else 0.0
    ap = _ap_from_pr(P, R)

    return dict(
        ODS=ods, ODS_threshold=ods_th, OIS=ois, AP=float(ap),
        Precision=ods_p, Recall=ods_r, num_images=int(len(preds))
    )

def maybe_profile(args, res: dict):
    if not args.ckpt:
        return
    try:
        from thop import profile
        from models.net import DMOREdgeNet
        dev = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
        model = DMOREdgeNet(channels=args.channels, topk=args.topk, router_mode=args.router_mode, temperature=args.temperature).to(dev).eval()
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)

        params = sum(p.numel() for p in model.parameters())
        res["Params_M"] = params / 1e6

        x = torch.randn(1, 3, args.img_size, args.img_size, device=dev)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        res["FLOPs_G"] = float(flops) / 1e9

        warmup, iters = 10, 50
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        res["FPS"] = iters / max(1e-9, (t1-t0))
    except Exception as e:
        print("⚠️ Complexity profiling skipped:", e)

def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    res = eval_pixel(pred_dir, gt_dir)
    maybe_profile(args, res)

    print("----------- Evaluation Result -----------")
    for k, v in res.items():
        print(f"{k}: {v}")

    save_dir = Path(args.save_dir) if args.save_dir else pred_dir.parent / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    outp = save_dir / "metrics.json"
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {outp}")

if __name__ == "__main__":
    main()
