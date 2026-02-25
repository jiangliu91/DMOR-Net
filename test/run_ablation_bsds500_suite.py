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
    repo_root = overlay_root.parent
    
    # 简化版注入：直接通过 sys.path 优先级让 Python 自己去加载 overlay 里的文件
    injection_code = f"""
import sys, os
from pathlib import Path

# 🟢 终极路径优先级：强制让 overlay/models 文件夹排在最前面
sys.path.insert(0, r'{repo_root}')
sys.path.insert(0, r'{overlay_root}')

import runpy
sys.argv = [r'{script_path}'] + {args_list}
print(f"--- Subprocess training start ---")
runpy.run_path(r'{script_path}', run_name='__main__')
"""
    return ["/root/miniconda3/envs/dmor/bin/python", "-c", injection_code]
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
    parser.add_argument("--num_workers", type=int, default=4)
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

    def run_one(tag: str, enabled_ops=None, pool_mode="dmor"):

        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        pred_dir = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        
        # 🟢 修复 1：明确指定 checkpoint 的完整路径
        ckpt_path = ckpt_dir / args.ckpt_name

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print("Running:", tag)
        print("Overlay used:", overlay)
        print("==============================")

        # ---------------- TRAIN ----------------
        train_args = [
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
            "--num_workers", str(args.num_workers),
            "--backbone", "lite",
            "--amp",
        ]
        
        if enabled_ops is not None:
            train_args.append("--enabled_ops")
            train_args.extend([str(op) for op in enabled_ops])
        train_args.extend(["--pool_mode", pool_mode])
        
        # 🟢 修复 2：实际构建命令并拉起训练进程
        train_script = overlay / "scripts" / "bsds_train.py"
        train_cmd = build_launch_command(train_script, overlay, train_args)
        print(f"[{tag}] Starting Training...")
        subprocess.run(train_cmd, check=True)

        # ---------------- EXPORT ----------------
        export_args = [
            "--input_dir", str(data_root / "images/test"),
            "--output_dir", str(pred_dir),
            "--checkpoint", str(ckpt_path),
            "--channels", str(args.channels),
            "--topk", str(args.topk),
            "--router_mode", args.router_mode,
            "--temperature", str(args.temperature),
        ]

        if enabled_ops is not None:
            export_args.append("--enabled_ops")
            export_args.extend([str(op) for op in enabled_ops])
        export_args.extend(["--pool_mode", pool_mode])

        # 🟢 修复：透传 --mst 参数，确保公平对比
        if args.mst:
            export_args.append("--mst")

        # 🟢 修复 3：实际构建命令并拉起导出进程
        export_script = overlay / "scripts" / "bsds_export.py"
        export_cmd = build_launch_command(export_script, overlay, export_args)
        print(f"[{tag}] Starting Export...")
        subprocess.run(export_cmd, check=True)

        # ---------------- EVAL ----------------
        eval_cmd = [
            "python",
            "eval_bsds500.py",
            "--pred_dir", str(pred_dir),
            "--gt_dir", str(data_root / "groundTruth/test"),
            "--device", args.device,
            "--save_dir", str(eval_dir),
        ]

        print(f"[{tag}] Starting Evaluation...")
        subprocess.run(eval_cmd, cwd=repo_root, check=True)

    experiments = [
        ("B1_noO1", [1,2,3,4], "dmor"),
        ("B2_noO2", [0,2,3,4], "dmor"),
        ("B3_noO3", [0,1,3,4], "dmor"),
        ("B4_noO4", [0,1,2,4], "dmor"),
        ("B5_noO5", [0,1,2,3], "dmor"),
        # ("B6_all3x3", None, "all3x3"),  # <-- 注释掉，保留战果
    ]

    for name, enabled, mode in experiments:
        run_one(
            tag=f"{args.exp_prefix}_{name}",
            enabled_ops=enabled,
            pool_mode=mode
        )

    print("\n✔ All ablations finished (overlay enforced).")

if __name__ == "__main__":
    main()
