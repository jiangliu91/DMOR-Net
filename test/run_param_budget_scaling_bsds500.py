"""Parameter-budget scaling experiment runner for DMOR-Edge on BSDS500.

This script runs a *set* of width variants (Tiny/Small/Main/Large) end-to-end:
train -> export test_png -> official GPU eval (ODS/OIS/AP + Params/FLOPs/FPS).

It is designed to match your existing repo layout:
  - train script:   scripts/bsds_train.py
  - export script:  scripts/bsds_export.py
  - eval script:    pipelines/eval_bsds500.py

Outputs are written under:
  {outputs_root}/BSDS500/{exp_prefix}_{variant}/
    ckpt/ , test_png/ , eval_official_gpu/

Usage example (Linux):
  python run_param_budget_scaling_bsds500.py \
    --repo_root /home/yuzhejia/DMOR/DMOR-Edge \
    --data_root /home/yuzhejia/DMOR/DMOR-Edge/dataset/BSDS500/data \
    --outputs_root /home/yuzhejia/DMOR/outputs/BSDS500 \
    --device cuda --epochs 100 --batch 4 --img_size 512
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def _read_metrics_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--data_root", type=str, required=True, help="BSDS500/data")
    ap.add_argument("--outputs_root", type=str, required=True, help=".../outputs/BSDS500")
    ap.add_argument("--exp_prefix", type=str, default="DMOR")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--router_mode", type=str, default="dmor")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--backbone", type=str, default="lite")
    ap.add_argument(
        "--variants",
        type=str,
        default="tiny,small,main,large",
        help="Comma-separated: tiny,small,main,large",
    )
    ap.add_argument(
        "--channels_map",
        type=str,
        default="tiny:16,small:24,main:32,large:48",
        help="Comma-separated mapping, e.g. tiny:16,small:24,main:32,large:48",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    # Parse channels map
    ch_map: Dict[str, int] = {}
    for item in args.channels_map.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        ch_map[k.strip().lower()] = int(v.strip())

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ch_map:
            raise ValueError(f"Variant '{v}' missing in --channels_map")

    summary: List[Dict] = []

    for v in variants:
        channels = ch_map[v]
        tag = f"{args.exp_prefix}_{v.capitalize()}"
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        test_png = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_path = ckpt_dir / "best.pth"

        print("\n" + "=" * 40)
        print(f"Running: {tag}")
        print(f"channels: {channels}")
        print(f"out_dir: {out_dir}")
        print("=" * 40, flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "bsds_train.py"),
                "--data_root",
                str(data_root),
                "--out_dir",
                str(out_dir),
                "--ckpt_dir",
                str(ckpt_dir),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch",
                str(args.batch),
                "--lr",
                str(args.lr),
                "--img_size",
                str(args.img_size),
                "--num_workers",
                str(args.num_workers),
                "--channels",
                str(channels),
                "--topk",
                str(args.topk),
                "--router_mode",
                args.router_mode,
                "--temperature",
                str(args.temperature),
                "--backbone",
                args.backbone,
                "--amp",
            ]
        )

        # 2) Export test_png
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "bsds_export.py"),
                "--data_root",
                str(data_root),
                "--ckpt",
                str(ckpt_path),
                "--out_dir",
                str(test_png),
                "--device",
                args.device,
                "--img_size",
                str(args.img_size),
                "--channels",
                str(channels),
                "--topk",
                str(args.topk),
                "--router_mode",
                args.router_mode,
                "--temperature",
                str(args.temperature),
                "--backbone",
                args.backbone,
            ]
        )

        # 3) Eval (ODS/OIS/AP + Params/FLOPs/FPS)
        _run(
            [
                sys.executable,
                str(repo_root / "pipelines" / "eval_bsds500.py"),
                "--pred_dir",
                str(test_png),
                "--gt_dir",
                str(data_root / "groundTruth" / "test"),
                "--device",
                args.device,
                "--save_dir",
                str(eval_dir),
                "--img_size",
                str(args.img_size),
                "--channels",
                str(channels),
                "--topk",
                str(args.topk),
                "--ckpt",
                str(ckpt_path),
            ]
        )

        m = _read_metrics_json(eval_dir / "metrics.json")
        m.update({"variant": v, "channels": channels, "tag": tag})
        summary.append(m)

    # Save summary
    out_summary = outputs_root / f"{args.exp_prefix}_param_budget_scaling_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to: {out_summary}")


if __name__ == "__main__":
    main()
