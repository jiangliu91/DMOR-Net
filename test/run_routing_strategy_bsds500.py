
# -*- coding: utf-8 -*-
"""
Clean Routing Strategy Experiment (Main Model Only)

✔ No overlay
✔ No PYTHONPATH hacks
✔ Uses main models/dmor.py
✔ Supports dmor / uniform / global / spatial
✔ Outputs Params / FLOPs / FPS
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
from thop import profile


def run_cmd(cmd: list[str], cwd: Path):
    print("\n[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def compute_efficiency(model, device, img_size=512, iters=50):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Params
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # FLOPs
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    flops = flops / 1e9

    # FPS
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        with torch.no_grad():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = iters / elapsed

    return params, flops, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--exp_prefix", default="DMOR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ckpt_name", default="dmor_best.pth")
    parser.add_argument("--mst", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    router_modes = ["uniform", "global", "spatial", "dmor"]

    for router_mode in router_modes:

        print("\n==============================")
        print("Running router_mode =", router_mode)
        print("==============================")

        out_dir = outputs_root / f"{args.exp_prefix}_Route{router_mode.capitalize()}"
        ckpt_dir = out_dir / "ckpt"
        pred_dir = out_dir / "test_png"
        eval_dir = out_dir / "eval"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Train
        cmd = [
            "python", "-m", "scripts.bsds_train",
            "--data_root", str(data_root),
            "--out_dir", str(out_dir),
            "--ckpt_dir", str(ckpt_dir),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--channels", str(args.channels),
            "--topk", str(args.topk),
            "--router_mode", router_mode,
            "--temperature", str(args.temperature),
            "--backbone", "lite",
            "--amp",
        ]
        run_cmd(cmd, cwd=repo_root)

        ckpt_path = ckpt_dir / args.ckpt_name
        if not ckpt_path.exists():
            print("[WARN] Checkpoint not found, skipping...")
            continue

        # Export
        cmd = [
            "python", "-m", "scripts.bsds_export",
            "--input_dir", str(data_root / "images/test"),
            "--output_dir", str(pred_dir),
            "--checkpoint", str(ckpt_path),
            "--channels", str(args.channels),
            "--topk", str(args.topk),
            "--router_mode", router_mode,
            "--temperature", str(args.temperature),
        ]
        if args.mst:
            cmd.append("--mst")
        run_cmd(cmd, cwd=repo_root)

        # Eval
        cmd = [
            "python", "eval_bsds500.py",
            "--pred_dir", str(pred_dir),
            "--gt_dir", str(data_root / "groundTruth/test"),
            "--device", args.device,
            "--save_dir", str(eval_dir),
        ]
        run_cmd(cmd, cwd=repo_root)

        # Efficiency
        sys.path.insert(0, str(repo_root))
        from models.net import DMOREdgeNet

        model = DMOREdgeNet(
            channels=args.channels,
            topk=args.topk,
            router_mode=router_mode,
            temperature=args.temperature,
            backbone="lite",
        ).to(device)

        params, flops, fps = compute_efficiency(model, device, args.img_size)

        print("\nEfficiency Results")
        print("Params (M): %.3f" % params)
        print("FLOPs (G): %.3f" % flops)
        print("FPS: %.2f" % fps)


if __name__ == "__main__":
    main()
