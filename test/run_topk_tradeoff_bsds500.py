"""
Top-K Sparsity Trade-off (BSDS500) runner for DMOR-Edge
=======================================================

Runs a suite over different Top-K settings and records:
  - Official metrics: ODS / OIS / AP (from your eval_bsds500.py GPU evaluator)
  - Efficiency: Params (M), FLOPs (G, if available), FPS (measured on your device)

This script is designed to match your "Routing mechanism analysis (Top-K)" table.

Default K list:
  K=5 (dense / 100%), K=3 (60%), K=2 (40%), K=1 (20%)

Notes:
- Your DMOR currently supports router_mode in {"dmor","uniform"}.
- "Dense" corresponds to K>=num_ops or K<=0. For a 5-op pool, K=5 => dense.
- FLOPs requires either `fvcore` or `thop`. If neither is installed, FLOPs will be "NA".

Usage (Windows example, run from repo root):
  python run_topk_tradeoff_bsds500.py ^
    --repo_root . ^
    --data_root .\dataset\BSDS500\data ^
    --outputs_root ..\outputs\BSDS500 ^
    --exp_prefix DMOR ^
    --device cuda ^
    --channels 32 --img_size 512 --epochs 100 --batch 4 ^
    --router_mode dmor --temperature 1.0 ^
    --mst
"""

from __future__ import annotations
import argparse
import json
import os
import time
import subprocess
from pathlib import Path

import torch


def run_cmd(cmd: list[str], env: dict, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def count_params(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n / 1e6


def estimate_flops_g(model: torch.nn.Module, x: torch.Tensor) -> str:
    # Try fvcore first
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total()
        return f"{flops/1e9:.3f}"
    except Exception:
        pass
    # Try thop
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return f"{flops/1e9:.3f}"
    except Exception:
        return "NA"


@torch.no_grad()
def measure_fps(model: torch.nn.Module, x: torch.Tensor, iters: int = 200, warmup: int = 50) -> float:
    device = x.device
    model.eval()

    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    sec = max(t1 - t0, 1e-9)
    return iters / sec


def load_metrics_json(eval_dir: Path) -> dict:
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--outputs_root", required=True)

    ap.add_argument("--exp_prefix", default="DMOR")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--router_mode", default="dmor")
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--k_list", default="5,3,2,1", help="comma-separated K list, e.g. 5,3,2,1")
    ap.add_argument("--ckpt_name", default="dmor_best.pth")
    ap.add_argument("--mst", action="store_true")

    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_train", action="store_true")
    ap.add_argument("--no_export", action="store_true")

    ap.add_argument("--fps_iters", type=int, default=200)
    ap.add_argument("--fps_warmup", type=int, default=50)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip() != ""]
    num_ops = 5  # fixed in your operator pool
    suite_tag = f"{args.exp_prefix}_TopKTradeoff"
    suite_dir = outputs_root / suite_tag
    suite_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    # Prepare summary containers
    rows = []

    # Import model for efficiency measurement from repo (no overlay needed here)
    # We set PYTHONPATH so "from models.net import DMOREdgeNet" works reliably.
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH','')}"
    from models.net import DMOREdgeNet  # type: ignore

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for k in k_list:
        active_ratio = min(max(k, 0), num_ops) / float(num_ops) if (0 < k < num_ops) else 1.0
        tag = f"{args.exp_prefix}_K{k}"
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        pred_dir = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print(f"Running: {tag}")
        print(f"Top-K: {k} | active_ratio={active_ratio*100:.1f}%")
        print(f"out_dir: {out_dir}")
        print("==============================")

        if not args.no_train:
            cmd = [
                "python", "scripts/bsds_train.py",
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
                "--topk", str(k),
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
                "--backbone", "lite",
                "--amp",
            ]
            run_cmd(cmd, env=env, cwd=repo_root)

        ckpt_path = ckpt_dir / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[WARN] ckpt not found: {ckpt_path} (skip export/eval/efficiency)")
            continue

        if not args.no_export:
            cmd = [
                "python", "scripts/bsds_export.py",
                "--input_dir", str(data_root / "images/test"),
                "--output_dir", str(pred_dir),
                "--checkpoint", str(ckpt_path),
                "--channels", str(args.channels),
                "--topk", str(k),
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
            ]
            if args.mst:
                cmd.append("--mst")
            run_cmd(cmd, env=env, cwd=repo_root)

        if not args.no_eval:
            cmd = [
                "python", "eval_bsds500.py",
                "--pred_dir", str(pred_dir),
                "--gt_dir", str(data_root / "groundTruth/test"),
                "--device", args.device,
                "--save_dir", str(eval_dir),
            ]
            run_cmd(cmd, env=env, cwd=repo_root)

        # Efficiency measurement
        model = DMOREdgeNet(
            channels=args.channels,
            topk=k,
            router_mode=args.router_mode,
            temperature=args.temperature,
            backbone="lite",
        ).to(device)
        sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)

        x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        params_m = count_params(model)
        flops_g = estimate_flops_g(model, x)
        fps = measure_fps(model, x, iters=args.fps_iters, warmup=args.fps_warmup)

        metrics = load_metrics_json(eval_dir)
        ods = metrics.get("ODS", None)
        ois = metrics.get("OIS", None)
        apv = metrics.get("AP", None)

        row = {
            "tag": tag,
            "topk": k,
            "active_ratio": active_ratio,
            "ODS": ods,
            "OIS": ois,
            "AP": apv,
            "Params(M)": round(params_m, 4),
            "FLOPs(G)": flops_g,
            "FPS": round(float(fps), 3),
            "ckpt": str(ckpt_path),
            "pred_dir": str(pred_dir),
            "eval_dir": str(eval_dir),
        }
        rows.append(row)

        # Save per-run efficiency json
        (out_dir / "efficiency.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    # Save summary
    summary_json = suite_dir / "summary_topk.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Also write a simple CSV
    summary_csv = suite_dir / "summary_topk.csv"
    if rows:
        cols = ["tag","topk","active_ratio","ODS","OIS","AP","Params(M)","FLOPs(G)","FPS","ckpt","pred_dir","eval_dir"]
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r.get(c,"")) for c in cols))
        summary_csv.write_text("\n".join(lines), encoding="utf-8")

    print("\n✅ Top-K tradeoff finished.")
    print("Summary JSON:", summary_json)
    print("Summary CSV :", summary_csv)


if __name__ == "__main__":
    main()
