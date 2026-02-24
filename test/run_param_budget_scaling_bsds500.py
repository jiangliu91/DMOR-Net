"""Parameter-budget scaling experiment runner for DMOR-Edge on BSDS500.

Optimized to prevent large-model overfitting by allowing per-variant hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str], env: Dict[str, str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def _read_metrics_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_map_arg(arg_str: str, value_type=float) -> Dict[str, any]:
    """Helper to parse comma-separated key:value maps (e.g., 'tiny:1e-3,main:5e-4')"""
    parsed_map = {}
    for item in arg_str.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        parsed_map[k.strip().lower()] = value_type(v.strip())
    return parsed_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--data_root", type=str, required=True, help="BSDS500/data")
    ap.add_argument("--outputs_root", type=str, required=True, help=".../outputs/BSDS500")
    ap.add_argument("--exp_prefix", type=str, default="DMOR")
    ap.add_argument("--device", type=str, default="cuda")
    
    # Base Hyperparams
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--router_mode", type=str, default="dmor")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--backbone", type=str, default="lite")
    
    # Scaling Maps (The Fix)
    ap.add_argument(
        "--variants", type=str, default="tiny,small,main,large",
        help="Comma-separated variants"
    )
    ap.add_argument(
        "--channels_map", type=str, default="tiny:16,small:24,main:32,large:48"
    )
    # Give larger models smaller learning rates or different epoch lengths
    ap.add_argument(
        "--lr_map", type=str, default="tiny:1e-3,small:1e-3,main:8e-4,large:5e-4"
    )
    ap.add_argument(
        "--epochs_map", type=str, default="tiny:100,small:100,main:90,large:80"
    )
    # [NEW] Give larger models HIGHER weight decay to explicitly prevent overfitting
    ap.add_argument(
        "--wd_map", type=str, default="tiny:1e-4,small:1e-4,main:2e-4,large:5e-4"
    )

    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    # Parse maps
    ch_map = parse_map_arg(args.channels_map, int)
    lr_map = parse_map_arg(args.lr_map, float)
    epochs_map = parse_map_arg(args.epochs_map, int)
    wd_map = parse_map_arg(args.wd_map, float) # [NEW]

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    
    # Validate maps
    for v in variants:
        if v not in ch_map: raise ValueError(f"Variant '{v}' missing in --channels_map")
        if v not in lr_map: raise ValueError(f"Variant '{v}' missing in --lr_map")
        if v not in epochs_map: raise ValueError(f"Variant '{v}' missing in --epochs_map")
        if v not in wd_map: raise ValueError(f"Variant '{v}' missing in --wd_map") # [NEW]

    summary: List[Dict] = []

    for v in variants:
        channels = ch_map[v]
        lr = lr_map[v]
        epochs = epochs_map[v]
        wd = wd_map[v] # [NEW]
        
        tag = f"{args.exp_prefix}_{v.capitalize()}"
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        test_png = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_path = ckpt_dir / "dmor_best.pth"

        print("\n" + "=" * 60)
        print(f"🚀 Running: {tag}")
        print(f"   Ch: {channels} | LR: {lr} | Ep: {epochs} | WD: {wd}")
        print("=" * 60, flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        train_cmd = [
            sys.executable, "-m", "scripts.bsds_train",
            "--data_root", str(data_root),
            "--out_dir", str(out_dir),
            "--ckpt_dir", str(ckpt_dir),
            "--device", args.device,
            "--epochs", str(epochs),       
            "--batch", str(args.batch),
            "--lr", str(lr),               
            "--weight_decay", str(wd),     # [NEW] Pass weight decay
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--channels", str(channels),
            "--topk", str(args.topk),
            "--router_mode", args.router_mode,
            "--temperature", str(args.temperature),
            "--backbone", args.backbone,
            "--amp",
        ]
        _run(train_cmd, env=os.environ.copy(), cwd=repo_root)

        # 2) Export test_png
        export_cmd = [
            sys.executable, "-m", "scripts.bsds_export",
            "--input_dir", str(data_root / "images" / "test"),
            "--output_dir", str(test_png),
            "--checkpoint", str(ckpt_path),
            "--channels", str(channels),
            "--topk", str(args.topk),
            "--router_mode", args.router_mode,
            "--temperature", str(args.temperature),
        ]
        _run(export_cmd, env=os.environ.copy(), cwd=repo_root)

        # 3) Eval
        eval_cmd = [
            sys.executable, "-m", "pipelines.eval_bsds500",
            "--pred_dir", str(test_png),
            "--gt_dir", str(data_root / "groundTruth" / "test"),
            "--device", args.device,
            "--save_dir", str(eval_dir),
            "--img_size", str(args.img_size),
            "--channels", str(channels),
            "--topk", str(args.topk),
            "--ckpt", str(ckpt_path),
        ]
        _run(eval_cmd, env=os.environ.copy(), cwd=repo_root)

        m = _read_metrics_json(eval_dir / "metrics.json")
        m.update({"variant": v, "channels": channels, "lr": lr, "epochs": epochs, "wd": wd, "tag": tag})
        summary.append(m)

    # Save summary
    out_summary = outputs_root / f"{args.exp_prefix}_param_budget_scaling_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ Saved summary to: {out_summary}")


if __name__ == "__main__":
    main()