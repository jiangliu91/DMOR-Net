"""
Routing Strategy Ablation (BSDS500) runner for DMOR-Edge
=======================================================

This is your "Operator Effectiveness / Routing Strategy" third ablation:
  - No Routing (Avg Fusion)      : uniform weights over operators
  - Global Only (SE-like)        : global router only (image-level weights, broadcast to pixels)
  - Spatial Only (Pixel-level)   : spatial router only (per-pixel weights)
  - Dual-Level (DMOR, Hybrid)    : global + spatial (your main)

Outputs per setting:
  - ODS / OIS / AP  (GPU eval via eval_bsds500.py)
  - Params(M), FLOPs(G if available), FPS (measured)

IMPORTANT:
- This script DOES NOT require editing your repo.
- It uses a temporary overlay `_overlay_dmor_router/` that overrides `models/dmor.py`
  to support router_mode in: uniform | global | spatial | dmor.

Usage (Windows, run from repo root):
  python run_routing_strategy_bsds500.py ^
    --repo_root . ^
    --data_root .\dataset\BSDS500\data ^
    --outputs_root ..\outputs\BSDS500 ^
    --exp_prefix DMOR ^
    --device cuda ^
    --channels 32 --img_size 512 --epochs 100 --batch 4 ^
    --topk 2 --temperature 1.0 ^
    --mst

Results:
  outputs_root/<exp_prefix>_RouteUniform
  outputs_root/<exp_prefix>_RouteGlobal
  outputs_root/<exp_prefix>_RouteSpatial
  outputs_root/<exp_prefix>_RouteDMOR
Summary:
  outputs_root/<exp_prefix>_RoutingStrategy/summary_routing.csv
  outputs_root/<exp_prefix>_RoutingStrategy/summary_routing.json
"""

from __future__ import annotations
import argparse
import json
import os
import time
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import torch


def run_cmd(cmd: list[str], env: dict, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def write_overlay(overlay_root: Path) -> None:
    """Overlay dmor.py to support router_mode: uniform/global/spatial/dmor."""
    models_dir = overlay_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Only dmor.py needs to be overridden; operators.py & net.py are imported from repo.
    (models_dir / "dmor.py").write_text(dedent(r"""
        from __future__ import annotations
        import torch
        import torch.nn as nn
        from dataclasses import dataclass

        from .operators import build_operator_pool


        @dataclass
        class RoutingConfig:
            router_mode: str = "dmor"   # uniform | global | spatial | dmor
            topk: int = 2               # only meaningful for "dmor" mode
            temperature: float = 1.0
            eps: float = 1e-6
            use_ste: bool = True


        class GlobalRouter(nn.Module):
            def __init__(self, channels: int, num_ops: int):
                super().__init__()
                hidden = max(4, channels // 4)
                self.net = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, hidden, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, num_ops, 1, bias=True),
                )

            def forward(self, x):
                return self.net(x)  # [B,N,1,1]


        class SpatialRouter(nn.Module):
            def __init__(self, channels: int, num_ops: int):
                super().__init__()
                self.conv = nn.Conv2d(channels, num_ops, 1, bias=True)

            def forward(self, x):
                return self.conv(x)  # [B,N,H,W]


        class DMOR(nn.Module):
            def __init__(
                self,
                channels: int,
                topk: int = 2,
                router_mode: str = "dmor",
                temperature: float = 1.0,
            ):
                super().__init__()
                self.channels = int(channels)
                self.cfg = RoutingConfig(router_mode=str(router_mode), topk=int(topk), temperature=float(temperature))

                # operator pool from repo (5 ops)
                self.ops = build_operator_pool(self.channels)
                self.num_ops = len(self.ops)
                if self.num_ops <= 0:
                    raise ValueError("Operator pool is empty.")

                # routers (create only what you need)
                if self.cfg.router_mode in ("dmor", "global"):
                    self.global_router = GlobalRouter(self.channels, self.num_ops)
                if self.cfg.router_mode in ("dmor", "spatial"):
                    self.spatial_router = SpatialRouter(self.channels, self.num_ops)

            def _uniform(self, x: torch.Tensor) -> torch.Tensor:
                b, _, h, w = x.shape
                return torch.full((b, self.num_ops, h, w), 1.0 / self.num_ops, device=x.device, dtype=x.dtype)

            def _compute_weights(self, x: torch.Tensor) -> torch.Tensor:
                mode = self.cfg.router_mode

                if mode == "uniform":
                    return self._uniform(x)

                if mode == "global":
                    g_logits = self.global_router(x)  # [B,N,1,1]
                    g_logits = g_logits / max(self.cfg.temperature, self.cfg.eps)
                    g = torch.softmax(g_logits, dim=1)  # [B,N,1,1]
                    return g.expand(-1, -1, x.shape[2], x.shape[3])  # broadcast to [B,N,H,W]

                if mode == "spatial":
                    s_logits = self.spatial_router(x)  # [B,N,H,W]
                    s_logits = s_logits / max(self.cfg.temperature, self.cfg.eps)
                    return torch.softmax(s_logits, dim=1)

                # mode == "dmor" (global + spatial)
                g_logits = self.global_router(x)      # [B,N,1,1]
                s_logits = self.spatial_router(x)     # [B,N,H,W]
                logits = (g_logits + s_logits) / max(self.cfg.temperature, self.cfg.eps)
                return torch.softmax(logits, dim=1)

            def _apply_topk(self, weights: torch.Tensor) -> torch.Tensor:
                # Only apply Top-K for "dmor" (hybrid), consistent with your definition.
                if self.cfg.router_mode != "dmor":
                    return weights
                if self.cfg.topk <= 0 or self.cfg.topk >= self.num_ops:
                    return weights

                _, topk_idx = torch.topk(weights, k=self.cfg.topk, dim=1)
                mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
                w_hard = weights * mask
                w_hard = w_hard / (w_hard.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps))

                if self.cfg.use_ste:
                    return w_hard - w_hard.detach() + weights
                return w_hard

            def forward(self, x: torch.Tensor, return_weights: bool = False):
                weights = self._compute_weights(x)
                weights = self._apply_topk(weights)

                out = x.new_zeros(x.shape)
                for i, op in enumerate(self.ops):
                    out = out + op(x) * weights[:, i:i+1, :, :]

                if return_weights:
                    return out, weights
                return out
    """).lstrip(), encoding="utf-8")


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops_g(model: torch.nn.Module, x: torch.Tensor) -> str:
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total()
        return f"{flops/1e9:.3f}"
    except Exception:
        pass
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return f"{flops/1e9:.3f}"
    except Exception:
        return "NA"


@torch.no_grad()
def measure_fps(model: torch.nn.Module, x: torch.Tensor, iters: int = 200, warmup: int = 50) -> float:
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    return iters / max(t1 - t0, 1e-9)


def load_metrics_json(eval_dir: Path) -> dict:
    p = eval_dir / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
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

    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)

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

    overlay = Path.cwd() / "_overlay_dmor_router"
    if overlay.exists():
        shutil.rmtree(overlay)
    write_overlay(overlay)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{overlay}:{repo_root}:{env.get('PYTHONPATH','')}"

    # Import model from repo with overlay dmor
    from models.net import DMOREdgeNet  # type: ignore

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    settings = [
        ("RouteUniform", "uniform"),
        ("RouteGlobal",  "global"),
        ("RouteSpatial", "spatial"),
        ("RouteDMOR",    "dmor"),
    ]

    rows = []
    suite_dir = outputs_root / f"{args.exp_prefix}_RoutingStrategy"
    suite_dir.mkdir(parents=True, exist_ok=True)

    for tag_suffix, router_mode in settings:
        tag = f"{args.exp_prefix}_{tag_suffix}"
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        pred_dir = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print("Running:", tag)
        print("router_mode:", router_mode)
        print("out_dir:", out_dir)
        print("==============================")

        # Train
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
                "--topk", str(args.topk),
                "--router_mode", router_mode,
                "--temperature", str(args.temperature),
                "--backbone", "lite",
                "--amp",
            ]
            run_cmd(cmd, env=env, cwd=repo_root)

        ckpt_path = ckpt_dir / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[WARN] ckpt not found: {ckpt_path} (skip export/eval/efficiency)")
            continue

        # Export
        if not args.no_export:
            cmd = [
                "python", "scripts/bsds_export.py",
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
            run_cmd(cmd, env=env, cwd=repo_root)

        # Eval
        if not args.no_eval:
            cmd = [
                "python", "eval_bsds500.py",
                "--pred_dir", str(pred_dir),
                "--gt_dir", str(data_root / "groundTruth/test"),
                "--device", args.device,
                "--save_dir", str(eval_dir),
            ]
            run_cmd(cmd, env=env, cwd=repo_root)

        # Efficiency
        model = DMOREdgeNet(
            channels=args.channels,
            topk=args.topk,
            router_mode=router_mode,
            temperature=args.temperature,
            backbone="lite",
        ).to(device)

        sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)

        x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        params_m = count_params_m(model)
        flops_g = estimate_flops_g(model, x)
        fps = measure_fps(model, x, iters=args.fps_iters, warmup=args.fps_warmup)

        metrics = load_metrics_json(eval_dir)
        row = {
            "tag": tag,
            "router_mode": router_mode,
            "ODS": metrics.get("ODS", None),
            "OIS": metrics.get("OIS", None),
            "AP": metrics.get("AP", None),
            "Params(M)": round(float(params_m), 4),
            "FLOPs(G)": flops_g,
            "FPS": round(float(fps), 3),
            "ckpt": str(ckpt_path),
            "pred_dir": str(pred_dir),
            "eval_dir": str(eval_dir),
        }
        rows.append(row)
        (out_dir / "efficiency.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    # Save summaries
    (suite_dir / "summary_routing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        cols = ["tag","router_mode","ODS","OIS","AP","Params(M)","FLOPs(G)","FPS","ckpt","pred_dir","eval_dir"]
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r.get(c,"")) for c in cols))
        (suite_dir / "summary_routing.csv").write_text("\n".join(lines), encoding="utf-8")

    print("\n✅ Routing strategy ablation finished.")
    print("Summary dir:", suite_dir)


if __name__ == "__main__":
    main()
