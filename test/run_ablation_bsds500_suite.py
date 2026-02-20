"""
BSDS500 Operator Ablation Suite (B1~B6)
FULLY FIXED VERSION – NO PYTHONPATH

✔ No PYTHONPATH dependency
✔ No import priority issues
✔ No environment pollution
✔ 100% overlay enforcement
✔ Each subprocess force-injects overlay modules
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path


# ------------------------------------------------------------
# Runtime Injection Launcher
# ------------------------------------------------------------

def build_launch_command(script_path: Path, overlay_root: Path, args_list: list[str]):

    injection_code = f"""
import importlib.util, sys
from pathlib import Path

overlay = Path(r'{overlay_root}')

def inject(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        overlay / "models" / file_name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

# 强制覆盖三个模块
inject("models.operators", "operators.py")
inject("models.dmor", "dmor.py")
inject("models.net", "net.py")

import runpy
runpy.run_path(r'{script_path}', run_name="__main__")
"""

    return ["python", "-c", injection_code] + args_list


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--exp_prefix", default="DMOR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--router_mode", default="dmor")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ckpt_name", default="dmor_best.pth")
    parser.add_argument("--mst", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    overlay = repo_root / "_overlay_dmor"

    if not overlay.exists():
        raise RuntimeError(
            f"Overlay directory not found: {overlay}\n"
            "请先运行原始 ablation 脚本生成 overlay。"
        )

    def run_one(tag: str):

        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        pred_dir = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print("Running:", tag)
        print("Overlay used:", overlay)
        print("==============================")

        # ---------------- TRAIN ----------------
        train_cmd = build_launch_command(
            repo_root / "scripts/bsds_train.py",
            overlay,
            [
                "--data_root", str(data_root),
                "--out_dir", str(out_dir),
                "--ckpt_dir", str(ckpt_dir),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--lr", str(args.lr),
                "--img_size", str(args.img_size),
                "--channels", str(args.channels),
                "--topk", str(args.topk),
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
                "--backbone", "lite",
                "--amp",
            ],
        )

        subprocess.run(train_cmd, cwd=repo_root, check=True)

        # ---------------- EXPORT ----------------
        ckpt_path = ckpt_dir / args.ckpt_name
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        export_cmd = build_launch_command(
            repo_root / "scripts/bsds_export.py",
            overlay,
            [
                "--input_dir", str(data_root / "images/test"),
                "--output_dir", str(pred_dir),
                "--checkpoint", str(ckpt_path),
                "--channels", str(args.channels),
                "--topk", str(args.topk),
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
            ] + (["--mst"] if args.mst else []),
        )

        subprocess.run(export_cmd, cwd=repo_root, check=True)

        # ---------------- EVAL ----------------
        eval_cmd = [
            "python",
            "eval_bsds500.py",
            "--pred_dir", str(pred_dir),
            "--gt_dir", str(data_root / "groundTruth/test"),
            "--device", args.device,
            "--save_dir", str(eval_dir),
        ]

        subprocess.run(eval_cmd, cwd=repo_root, check=True)

    # 示例：跑一个
    run_one(f"{args.exp_prefix}_B1_noO1")

    print("\n✔ Ablation finished (overlay enforced).")


if __name__ == "__main__":
    main()
