"""Alpha sensitivity experiment runner for DMOR-Edge on BSDS500.

We treat alpha as the mixing weight between global-router logits and spatial-router logits:
  logits = alpha * global_logits + (1 - alpha) * spatial_logits

This file **does not require editing your repo code**. It creates a small overlay
that overrides `models/dmor.py` and reads `DMOR_ALPHA` from environment.

Runs: train -> export -> eval for alpha in {0.0, 0.3, 0.5, 0.7, 1.0} (customizable).

Usage example:
  python run_alpha_sensitivity_bsds500.py \
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


_PATCHED_DMOR_PY = r"""# Auto-generated overlay for DMOR alpha sensitivity.
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import make_operator_pool


@dataclass
class RoutingConfig:
    topk: int = 2
    temperature: float = 1.0
    use_global: bool = True
    use_spatial: bool = True
    alpha: float = 0.5  # global weight; (1-alpha) is spatial weight


class DMOR(nn.Module):
    def __init__(
        self,
        channels: int,
        topk: int = 2,
        temperature: float = 1.0,
        pool_mode: str = "dmor",
    ):
        super().__init__()
        self.cfg = RoutingConfig(topk=topk, temperature=temperature)
        self.channels = channels

        # operator pool
        self.ops, self.num_ops = make_operator_pool(channels, mode=pool_mode)

        # routers
        self.global_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.num_ops, 1),
        )
        self.spatial_router = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.num_ops, 1),
        )

    def _get_alpha(self) -> float:
        env_a = os.getenv("DMOR_ALPHA", "").strip()
        if env_a:
            try:
                return float(env_a)
            except Exception:
                return float(self.cfg.alpha)
        return float(self.cfg.alpha)

    def _topk_mask(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        # logits: [B, N, H, W]
        if k <= 0 or k >= logits.shape[1]:
            return torch.ones_like(logits, dtype=torch.bool)
        topk_idx = torch.topk(logits, k=k, dim=1).indices  # [B,k,H,W]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute operator features
        op_feats = []
        for op in self.ops:
            op_feats.append(op(x))
        op_feats = torch.stack(op_feats, dim=1)  # [B,N,C,H,W]

        # routing logits
        s_logits = self.spatial_router(x)  # [B,N,H,W]
        g_logits = self.global_router(x)   # [B,N,1,1]
        g_logits = g_logits.expand_as(s_logits)

        a = max(0.0, min(1.0, self._get_alpha()))
        logits = a * g_logits + (1.0 - a) * s_logits

        # temperature
        temp = max(1e-6, float(self.cfg.temperature))
        logits = logits / temp

        # top-k
        k = int(self.cfg.topk)
        mask = self._topk_mask(logits, k)
        logits = logits.masked_fill(~mask, float("-inf"))

        weights = F.softmax(logits, dim=1)  # [B,N,H,W]

        out = (weights.unsqueeze(2) * op_feats).sum(dim=1)
        return out
"""


def _ensure_overlay(repo_root: Path) -> Path:
    overlay_root = repo_root / "_overlay_dmor_alpha"
    models_dir = overlay_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "__init__.py").write_text("\n", encoding="utf-8")
    (models_dir / "dmor.py").write_text(_PATCHED_DMOR_PY, encoding="utf-8")
    return overlay_root


def _run(cmd: List[str], env: Dict[str, str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env, cwd=str(cwd))


def _read_metrics_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--outputs_root", type=str, required=True)
    ap.add_argument("--exp_prefix", type=str, default="DMOR_alpha")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--router_mode", type=str, default="dmor")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--backbone", type=str, default="lite")
    ap.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.3,0.5,0.7,1.0",
        help="Comma-separated alphas",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    overlay_root = _ensure_overlay(repo_root)

    base_env = os.environ.copy()
    # Prepend overlay + repo_root to PYTHONPATH
    py_path = os.pathsep.join([str(overlay_root), str(repo_root), base_env.get("PYTHONPATH", "")])
    base_env["PYTHONPATH"] = py_path

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    summary: List[Dict] = []

    for a in alphas:
        env = base_env.copy()
        env["DMOR_ALPHA"] = str(a)

        tag = f"{args.exp_prefix}_a{a:g}".replace(".", "p")
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        test_png = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_path = ckpt_dir / "dmor_best.pth"

        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"Running: {tag}")
        print("=" * 60)

        # Train
        _run(
            [
                sys.executable,
                "-m", "scripts.bsds_train",
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
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
                "--backbone", args.backbone,
                "--amp",
            ],
            env=env,
            cwd=repo_root,
        )

        # Export
        _run(
            [
                sys.executable,
                "-m", "scripts.bsds_export",
                "--input_dir", str(data_root / "images" / "test"),
                "--output_dir", str(test_png),
                "--checkpoint", str(ckpt_path),
                "--channels", str(args.channels),
                "--topk", str(args.topk),
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
            ],
            env=env,
            cwd=repo_root,
        )

        # Eval
        _run(
            [
                sys.executable,
                "-m", "pipelines.eval_bsds500",
                "--pred_dir", str(test_png),
                "--gt_dir", str(data_root / "groundTruth" / "test"),
                "--device", args.device,
                "--save_dir", str(eval_dir),
                "--ckpt", str(ckpt_path),
                "--img_size", str(args.img_size),
                "--channels", str(args.channels),
                "--topk", str(args.topk),
            ],
            env=env,
            cwd=repo_root,
        )

        m = _read_metrics_json(eval_dir / "metrics.json")
        m.update({"alpha": a})
        summary.append(m)

    summary_path = outputs_root / f"{args.exp_prefix}_alpha_sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to: {summary_path}")

if __name__ == "__main__":
    main()
