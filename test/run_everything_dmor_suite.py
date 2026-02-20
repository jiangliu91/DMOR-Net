# -*- coding: utf-8 -*-
"""
DMOR-Edge: ONE master entry to run *everything* (all datasets + all pipelines + all BSDS500 experiment suites)

# ================= Added: Auto Checkpoint Resolver =================
def _find_ckpt_auto(ckpt_dir, default_name):
    ckpt_dir = Path(ckpt_dir)
    p = ckpt_dir / default_name
    if p.is_file():
        return p
    cands = sorted(ckpt_dir.glob("*_best.pth"))
    if not cands:
        cands = sorted(ckpt_dir.glob("*.pth"))
    if not cands:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
    return cands[-1]
# ====================================================================



Fixes vs v1:
- eval scripts are searched in BOTH scripts/ and pipelines/ (your screenshots show pipelines/eval_*.py)
- creates outputs/_RUN_LOGS before you use `tee` (so "No such file or directory" won't happen)
- still keeps strict fail-fast behavior for reproducibility

Recommended screen command:
  cd /home/yuzhejia/DMOR/DMOR-Edge
  mkdir -p /home/yuzhejia/DMOR/outputs/_RUN_LOGS
  screen -S dmor_all
  python -u run_everything_dmor_suite_v2.py --device cuda --epochs 100 --batch 4 --img_size 512 --mst --amp \
    2>&1 | tee -a /home/yuzhejia/DMOR/outputs/_RUN_LOGS/dmor_all.log

Detach: Ctrl+A then D
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_cmd(cmd: list[str], cwd: Path):
    print("\n" + "=" * 100)
    print("[CMD]", " ".join(cmd))
    print("[CWD]", str(cwd))
    print("=" * 100, flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"[ERROR] command failed (exit={r.returncode}): {' '.join(cmd)}")


def find_file(repo_root: Path, rel_candidates: list[str]) -> Path:
    for rel in rel_candidates:
        p = (repo_root / rel).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError("Cannot find file. Tried:\n  - " + "\n  - ".join(str(repo_root / x) for x in rel_candidates))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser("DMOR-Edge master runner (all datasets + all pipelines)")
    # Paths
    p.add_argument("--repo_root", default="/home/yuzhejia/DMOR/DMOR-Edge")
    p.add_argument("--outputs_root", default="/home/yuzhejia/DMOR/outputs")
    p.add_argument("--dataset_root", default="", help="optional override; default uses {repo_root}/dataset")

    # Common training/export params
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="enable AMP if supported by train scripts")
    p.add_argument("--mst", action="store_true", help="enable multi-scale+flip TTA for export where supported")

    # Model params (shared)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--router_mode", default="dmor", choices=["dmor", "uniform"])
    p.add_argument("--temperature", type=float, default=1.0)

    # Checkpoint name convention
    p.add_argument("--ckpt_name", default="dmor_best.pth")

    # Dataset-specific path overrides (only needed if your folder names differ)
    p.add_argument("--bsds_data_dir", default="", help="BSDS500/data (contains images/ and groundTruth/)")
    p.add_argument("--biped_root", default="", help="BIPEDv2 root folder")
    p.add_argument("--biped_test_dir", default="", help="BIPEDv2 test images dir (default {biped_root}/test)")
    p.add_argument("--biped_gt_dir", default="", help="BIPEDv2 test gt dir (default {biped_root}/test_gt)")
    p.add_argument("--nyu_root", default="", help="NYUDv2 root folder")
    p.add_argument("--nyu_test_dir", default="", help="NYUDv2 test images dir (default {nyu_root}/test)")
    p.add_argument("--nyu_gt_dir", default="", help="NYUDv2 test gt dir (default {nyu_root}/test_gt)")

    # Toggles
    p.add_argument("--skip_bsds_pipeline", action="store_true")
    p.add_argument("--skip_bsds_suites", action="store_true")
    p.add_argument("--skip_biped", action="store_true")
    p.add_argument("--skip_nyudv2", action="store_true")

    # Optional forwarding to BSDS suites
    p.add_argument("--k_list", default="")
    p.add_argument("--alphas", default="")
    p.add_argument("--variants", default="")
    p.add_argument("--channels_map", default="")

    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else (repo_root / "dataset").resolve()
    py = sys.executable

    # Create log dir early (fix your tee error)
    ensure_dir(outputs_root / "_RUN_LOGS")

    # Resolve dataset roots
    bsds_data = Path(args.bsds_data_dir).resolve() if args.bsds_data_dir else (dataset_root / "BSDS500" / "data").resolve()
    biped_root = Path(args.biped_root).resolve() if args.biped_root else (dataset_root / "BIPEDv2").resolve()
    nyu_root = Path(args.nyu_root).resolve() if args.nyu_root else (dataset_root / "NYUDv2").resolve()

    # Outputs per dataset (match your screenshots)
    out_bsds = (outputs_root / "BSDS500" / "DMOR").resolve()
    out_biped = (outputs_root / "BIPEDv2" / "DMOR").resolve()
    out_nyu = (outputs_root / "NYUDv2" / "DMOR").resolve()
    for o in [out_bsds, out_biped, out_nyu]:
        ensure_dir(o / "ckpt")
        ensure_dir(o / "test_png")
        ensure_dir(o / "eval_official_gpu")

    # Locate scripts (train/export are in scripts/)
    f_bsds_train = find_file(repo_root, ["scripts/bsds_train.py", "bsds_train.py"])
    f_bsds_export = find_file(repo_root, ["scripts/bsds_export.py", "bsds_export.py"])
    f_biped_train = find_file(repo_root, ["scripts/biped_train.py", "biped_train.py"])
    f_biped_export = find_file(repo_root, ["scripts/biped_export.py", "biped_export.py"])
    f_nyu_train = find_file(repo_root, ["scripts/nyudv2_train.py", "nyudv2_train.py"])
    f_nyu_export = find_file(repo_root, ["scripts/nyudv2_export.py", "nyudv2_export.py"])

    # Locate eval scripts (your repo has them in pipelines/)
    f_eval_bsds = find_file(repo_root, ["pipelines/eval_bsds500.py", "scripts/eval_bsds500.py", "eval_bsds500.py"])
    f_eval_biped = find_file(repo_root, ["pipelines/eval_biped.py", "scripts/eval_biped.py", "eval_biped.py"])
    f_eval_nyu = find_file(repo_root, ["pipelines/eval_nyudv2.py", "scripts/eval_nyudv2.py", "eval_nyudv2.py"])

    # BSDS suite runners are in repo root (as shown)
    f_topk = find_file(repo_root, ["run_topk_tradeoff_bsds500.py"])
    f_routing = find_file(repo_root, ["run_routing_strategy_bsds500.py"])
    f_alpha = find_file(repo_root, ["run_alpha_sensitivity_bsds500.py"])
    f_budget = find_file(repo_root, ["run_param_budget_scaling_bsds500.py"])
    f_ablation = find_file(repo_root, ["run_ablation_bsds500_suite.py"])

    # BSDS pipeline (optional)
    f_bsds_pipeline = (repo_root / "pipelines" / "run_bsds500_pipeline.py").resolve()
    has_bsds_pipeline = f_bsds_pipeline.is_file()

    # Common CLI fragments
    common_model = [
        "--channels", str(args.channels),
        "--topk", str(args.topk),
        "--router_mode", str(args.router_mode),
        "--temperature", str(args.temperature),
    ]
    common_train = [
        "--device", str(args.device),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--img_size", str(args.img_size),
        "--num_workers", str(args.num_workers),
    ]

    print("[INFO] repo_root   =", repo_root)
    print("[INFO] dataset_root=", dataset_root)
    print("[INFO] outputs_root=", outputs_root)
    print("[INFO] BSDS data   =", bsds_data)
    print("[INFO] BIPED root  =", biped_root)
    print("[INFO] NYU root    =", nyu_root)
    print("[INFO] Using python=", py, flush=True)

    # ------------------------------------------------------------------------------------------------
    # [A] BSDS500 pipeline (your unified pipeline)
    # ------------------------------------------------------------------------------------------------
    if not args.skip_bsds_pipeline:
        if has_bsds_pipeline:
            cmd = [py, "-u", str(f_bsds_pipeline),
                   "--data_root", str(bsds_data),
                   "--out_root", str(out_bsds),
                   "--device", str(args.device),
                   "--img_size", str(args.img_size)]
            run_cmd(cmd, cwd=repo_root)
        else:
            print("[WARN] pipelines/run_bsds500_pipeline.py not found -> skip pipeline step.", flush=True)

    # ------------------------------------------------------------------------------------------------
    # [B] BSDS500 experiment suites (uploaded runners)
    # ------------------------------------------------------------------------------------------------
    if not args.skip_bsds_suites:
        suite_common = [
            "--repo_root", str(repo_root),
            "--data_root", str(bsds_data),
            "--outputs_root", str(outputs_root / "BSDS500"),
            "--exp_prefix", "DMOR",
            "--device", str(args.device),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--ckpt_name", str(args.ckpt_name),
        ] + common_model

        cmd = [py, "-u", str(f_topk)] + [x for i,x in enumerate(suite_common) if not (x=="--topk" or (i>0 and suite_common[i-1]=="--topk"))]
        if args.k_list.strip():
            cmd += ["--k_list", args.k_list.strip()]
        if args.mst:
            cmd += ["--mst"]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_routing)] + [
            "--repo_root", str(repo_root),
            "--data_root", str(bsds_data),
            "--outputs_root", str(outputs_root / "BSDS500"),
            "--exp_prefix", "DMOR",
            "--device", str(args.device),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--channels", str(args.channels),
            "--topk", str(args.topk),
            "--temperature", str(args.temperature),
            "--ckpt_name", str(args.ckpt_name),
        ]
        if args.mst:
            cmd += ["--mst"]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_alpha)] + [
            "--repo_root", str(repo_root),
            "--data_root", str(bsds_data),
            "--outputs_root", str(outputs_root / "BSDS500"),
            "--exp_prefix", "DMOR",
            "--device", str(args.device),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--router_mode", str(args.router_mode),
            "--topk", str(args.topk),
            "--temperature", str(args.temperature),
            "--channels", str(args.channels),
            "--backbone", "lite",
        ]
        if args.alphas.strip():
            cmd += ["--alphas", args.alphas.strip()]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_budget)] + [
            "--repo_root", str(repo_root),
            "--data_root", str(bsds_data),
            "--outputs_root", str(outputs_root / "BSDS500"),
            "--exp_prefix", "DMOR",
            "--device", str(args.device),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--img_size", str(args.img_size),
            "--num_workers", str(args.num_workers),
            "--router_mode", str(args.router_mode),
            "--topk", str(args.topk),
            "--temperature", str(args.temperature),
            "--backbone", "lite",
        ]
        if args.variants.strip():
            cmd += ["--variants", args.variants.strip()]
        if args.channels_map.strip():
            cmd += ["--channels_map", args.channels_map.strip()]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_ablation)] + suite_common
        if args.mst:
            cmd += ["--mst"]
        run_cmd(cmd, cwd=repo_root)

    # ------------------------------------------------------------------------------------------------
    # [C] BIPEDv2: train -> export -> eval
    # ------------------------------------------------------------------------------------------------
    if not args.skip_biped:
        biped_test = Path(args.biped_test_dir).resolve() if args.biped_test_dir else (biped_root / "test").resolve()
        biped_gt = Path(args.biped_gt_dir).resolve() if args.biped_gt_dir else (biped_root / "test_gt").resolve()

        cmd = [py, "-u", str(f_biped_train),
               "--data_root", str(biped_root),
               "--out_dir", str(out_biped),
               "--ckpt_dir", str(out_biped / "ckpt")] + common_train + common_model
        if args.amp:
            cmd += ["--amp"]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_biped_export),
               "--input_dir", str(biped_test),
               "--output_dir", str(out_biped / "test_png"),
               "--checkpoint", str(_find_ckpt_auto(out_biped / "ckpt", args.ckpt_name))] + common_model
        if args.mst:
            cmd += ["--mst"]
        cmd += ["--stretch"]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_eval_biped),
               "--pred_dir", str(out_biped / "test_png"),
               "--gt_dir", str(biped_gt),
               "--device", str(args.device),
               "--save_dir", str(out_biped / "eval_official_gpu")]
        run_cmd(cmd, cwd=repo_root)

    # ------------------------------------------------------------------------------------------------
    # [D] NYUDv2: train -> export -> eval
    # ------------------------------------------------------------------------------------------------
    if not args.skip_nyudv2:
        nyu_test = Path(args.nyu_test_dir).resolve() if args.nyu_test_dir else (nyu_root / "test").resolve()
        nyu_gt = Path(args.nyu_gt_dir).resolve() if args.nyu_gt_dir else (nyu_root / "test_gt").resolve()

        cmd = [py, "-u", str(f_nyu_train),
               "--data_root", str(nyu_root),
               "--out_dir", str(out_nyu),
               "--ckpt_dir", str(out_nyu / "ckpt")] + common_train + common_model
        if args.amp:
            cmd += ["--amp"]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_nyu_export),
               "--input_dir", str(nyu_test),
               "--output_dir", str(out_nyu / "test_png"),
               "--checkpoint", str(_find_ckpt_auto(out_nyu / "ckpt", args.ckpt_name))] + common_model
        if args.mst:
            cmd += ["--mst"]
        cmd += ["--stretch"]
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_eval_nyu),
               "--pred_dir", str(out_nyu / "test_png"),
               "--gt_dir", str(nyu_gt),
               "--device", str(args.device),
               "--save_dir", str(out_nyu / "eval_official_gpu")]
        run_cmd(cmd, cwd=repo_root)

    # Stamp
    stamp = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "dataset_root": str(dataset_root),
        "outputs_root": str(outputs_root),
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "img_size": args.img_size,
        "channels": args.channels,
        "topk": args.topk,
        "router_mode": args.router_mode,
        "temperature": args.temperature,
        "ran": {
            "bsds_pipeline": (not args.skip_bsds_pipeline) and has_bsds_pipeline,
            "bsds_suites": (not args.skip_bsds_suites),
            "biped": (not args.skip_biped),
            "nyudv2": (not args.skip_nyudv2),
        }
    }
    (outputs_root / "_RUN_LOGS" / "run_everything_stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")

    print("\n[OK] ALL DONE.")
    print(f"[STAMP] {(outputs_root / '_RUN_LOGS' / 'run_everything_stamp.json')}")


if __name__ == "__main__":
    main()



# ================= Added: NYUD Multi-Input (RGB/HHA/FUSION) =================
try:
    ckpt_nyu_auto = _find_ckpt_auto(out_nyu / "ckpt", args.ckpt_name)

    for _mode in ["rgb", "hha", "fusion"]:
        extra_out = out_nyu / _mode
        (extra_out / "test_png").mkdir(parents=True, exist_ok=True)
        (extra_out / "eval").mkdir(parents=True, exist_ok=True)

        cmd = [py, "-u", str(f_nyu_export),
               "--input_dir", str(nyu_test),
               "--output_dir", str(extra_out / "test_png"),
               "--checkpoint", str(ckpt_nyu_auto),
               "--input", _mode] + common_model
        run_cmd(cmd, cwd=repo_root)

        cmd = [py, "-u", str(f_eval_nyu),
               "--pred_dir", str(extra_out / "test_png"),
               "--gt_dir", str(nyu_gt),
               "--device", str(args.device),
               "--save_dir", str(extra_out / "eval")]
        run_cmd(cmd, cwd=repo_root)

except Exception as e:
    print("[NYUD multi-input extension skipped]", e)
# ============================================================================
