"""Parameter-budget scaling experiment runner for DMOR-Edge on BSDS500."""

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
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--outputs_root", type=str, required=True)
    ap.add_argument("--exp_prefix", type=str, default="DMOR")
    ap.add_argument("--device", type=str, default="cuda")
    
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--router_mode", type=str, default="dmor")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--backbone", type=str, default="lite")
    
    ap.add_argument("--variants", type=str, default="tiny,small,main,large")
    ap.add_argument("--channels_map", type=str, default="tiny:16,small:24,main:32,large:48")
    
    # 找回平衡：适中的学习率、轮数和权重衰减
    ap.add_argument("--lr_map", type=str, default="tiny:1e-3,small:1e-3,main:8e-4,large:4e-4")
    ap.add_argument("--epochs_map", type=str, default="tiny:100,small:100,main:90,large:80")
    ap.add_argument("--wd_map", type=str, default="tiny:1e-4,small:1e-4,main:2e-4,large:4e-4")
    
    # [NEW] 解决大通道下 DMOR 路由坍塌的核心策略：提高大模型的 Softmax 温度
    ap.add_argument("--temperature_map", type=str, default="tiny:1.0,small:1.0,main:1.2,large:2.5")

    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    ch_map = parse_map_arg(args.channels_map, int)
    lr_map = parse_map_arg(args.lr_map, float)
    epochs_map = parse_map_arg(args.epochs_map, int)
    wd_map = parse_map_arg(args.wd_map, float)
    temp_map = parse_map_arg(args.temperature_map, float) # [NEW]

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]

    summary: List[Dict] = []

    for v in variants:
        channels = ch_map[v]
        lr = lr_map[v]
        epochs = epochs_map[v]
        wd = wd_map[v]
        temp = temp_map[v] # [NEW]
        
        tag = f"{args.exp_prefix}_{v.capitalize()}"
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        test_png = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_path = ckpt_dir / "dmor_best.pth"

        print("\n" + "=" * 70)
        print(f"🚀 Running: {tag}")
        print(f"   Ch:{channels} | LR:{lr} | Ep:{epochs} | WD:{wd} | Temp:{temp}")
        print("=" * 70, flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, "-m", "scripts.bsds_train",
            "--data_root", str(data_root),
            "--out_dir", str(out_dir),
            "--ckpt_dir", str(ckpt_dir),
            "--device", args.device,
            "--epochs", str(epochs),       
            "--batch", str(args.batch),
            "--lr", str(lr),               
            "--weight_decay", str(wd),     
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--channels", str(channels),
            "--topk", str(args.topk),
            "--router_mode", args.router_mode,
            "--temperature", str(temp),    # [NEW] 传入动态温度
            "--backbone", args.backbone,
            "--amp",
        ]
        _run(train_cmd, env=os.environ.copy(), cwd=repo_root)

        export_cmd = [
            sys.executable, "-m", "scripts.bsds_export",
            "--input_dir", str(data_root / "images" / "test"),
            "--output_dir", str(test_png),
            "--checkpoint", str(ckpt_path),
            "--channels", str(channels),
            "--topk", str(args.topk),
            "--router_mode", args.router_mode,
            "--temperature", str(temp),    # [NEW] 测试时保持一致的温度
        ]
        _run(export_cmd, env=os.environ.copy(), cwd=repo_root)

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
        m.update({"variant": v, "channels": channels, "lr": lr, "epochs": epochs, "wd": wd, "temp": temp, "tag": tag})
        summary.append(m)

    out_summary = outputs_root / f"{args.exp_prefix}_param_budget_scaling_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ Saved summary to: {out_summary}")

if __name__ == "__main__":
    main()