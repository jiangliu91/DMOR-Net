"""
One-shot BSDS500 operator ablation suite (B1~B6) WITHOUT editing your repo files.

Key idea:
- We *overlay* patched modules (models/operators.py, models/dmor.py, models/net.py)
  into a temporary folder `_overlay_dmor/`.
- We then run your existing scripts via subprocess with:
      PYTHONPATH=_overlay_dmor:<repo_root>:$PYTHONPATH
  so imports prefer the overlay versions first.

This avoids modifying your model/scripts on disk, while still enabling:
  - enabled_ops (drop specific operators)
  - pool_mode=all3x3 (Replace All with 3×3 Conv baseline)

Usage example:
  python run_bsds500_ablation_suite_no_edit.py \
    --repo_root /home/yuzhejia/DMOR/DMOR-Edge \
    --data_root /home/yuzhejia/DMOR/DMOR-Edge/dataset/BSDS500/data \
    --outputs_root /home/yuzhejia/DMOR/outputs/BSDS500 \
    --exp_prefix DMOR \
    --device cuda \
    --channels 32 --img_size 512 --epochs 100 --batch 4 \
    --router_mode dmor --topk 2 --temperature 1.0 \
    --mst

Notes:
- Assumes best ckpt: {out_dir}/ckpt/dmor_best.pth (override with --ckpt_name)
- Requires in repo_root:
    scripts/bsds_train.py
    scripts/bsds_export.py
    eval_bsds500.py
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent


OPERATOR_ORDER = [
    "0 LearnableDiff",
    "1 CenterDiffConv (CDC)",
    "2 DirectionAware",
    "3 DilatedContext",
    "4 EdgePreserveSmooth",
]


def write_overlay(overlay_root: Path) -> None:
    models_dir = overlay_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # -------- operators.py --------
    (models_dir / "operators.py").write_text(dedent(r"""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


        class LearnableDiff(nn.Module):
            \"\"\"O1: x - DWConv(x)\"\"\"
            def __init__(self, in_channels: int):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
                nn.init.constant_(self.conv.weight, 1.0 / 9.0)

            def forward(self, x):
                return x - self.conv(x)


        class CenterDiffConv(nn.Module):
            \"\"\"O2: CDC depthwise 3x3\"\"\"
            def __init__(self, in_channels: int):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

            def forward(self, x):
                w = self.conv.weight  # [C,1,3,3]
                sum_w = w.sum(dim=(2, 3), keepdim=True)
                center = w[:, :, 1:2, 1:2]
                diff_w = w.clone()
                diff_w[:, :, 1:2, 1:2] = center - (sum_w - center)
                return F.conv2d(x, diff_w, None, stride=1, padding=1, dilation=1, groups=self.conv.groups)


        class DirectionAware(nn.Module):
            \"\"\"O3: horizontal + vertical\"\"\"
            def __init__(self, in_channels: int):
                super().__init__()
                self.h = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels, bias=False)
                self.v = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels, bias=False)

            def forward(self, x):
                return self.h(x) + self.v(x)


        class DilatedContext(nn.Module):
            \"\"\"O4: d2 + d3 context\"\"\"
            def __init__(self, in_channels: int):
                super().__init__()
                self.d2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels, bias=False)
                self.d3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, groups=in_channels, bias=False)

            def forward(self, x):
                return self.d2(x) + self.d3(x)


        class EdgePreserveSmooth(nn.Module):
            \"\"\"O5: x - avgpool(x)\"\"\"
            def __init__(self, in_channels: int):
                super().__init__()
                self.avg = nn.AvgPool2d(3, stride=1, padding=1)

            def forward(self, x):
                return x - self.avg(x)


        class PlainDW3x3(nn.Module):
            \"\"\"Control baseline: plain 3x3 depthwise conv (no edge prior).\"\"\"
            def __init__(self, in_channels: int):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

            def forward(self, x):
                return self.conv(x)


        def build_operator_pool(channels: int, enabled_ops: list[int] | None = None, pool_mode: str = "dmor") -> nn.ModuleList:
            \"\"\"Build operator pool.

            pool_mode:
              - "dmor": designed 5 operators (O1~O5)
              - "all3x3": replace all 5 operators with PlainDW3x3 (B6)
            enabled_ops:
              - None/empty: keep all
              - list of indices (0..4): keep selected
            \"\"\"
            channels = int(channels)

            if pool_mode == "all3x3":
                ops = [PlainDW3x3(channels) for _ in range(5)]
            else:
                ops = [
                    LearnableDiff(channels),
                    CenterDiffConv(channels),
                    DirectionAware(channels),
                    DilatedContext(channels),
                    EdgePreserveSmooth(channels),
                ]

            if enabled_ops is not None and len(enabled_ops) > 0:
                enabled_ops = [int(i) for i in enabled_ops]
                enabled_set = set(enabled_ops)
                ops = [op for idx, op in enumerate(ops) if idx in enabled_set]

            return nn.ModuleList(ops)
    """).lstrip(), encoding="utf-8")

    # -------- dmor.py --------
    (models_dir / "dmor.py").write_text(dedent(r"""
        from __future__ import annotations
        import torch
        import torch.nn as nn
        from dataclasses import dataclass
        from .operators import build_operator_pool


        @dataclass
        class RoutingConfig:
            router_mode: str = "dmor"   # "dmor" or "uniform"
            topk: int = 2               # 0 => dense
            temperature: float = 1.0
            eps: float = 1e-6
            use_ste: bool = True        # straight-through for Top-K


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
                enabled_ops: list[int] | None = None,
                pool_mode: str = "dmor",
            ):
                super().__init__()
                self.channels = int(channels)
                self.cfg = RoutingConfig(router_mode=router_mode, topk=int(topk), temperature=float(temperature))

                self.ops = build_operator_pool(self.channels, enabled_ops=enabled_ops, pool_mode=pool_mode)
                self.num_ops = len(self.ops)
                if self.num_ops <= 0:
                    raise ValueError("Operator pool is empty (check enabled_ops).")

                if self.cfg.router_mode == "dmor":
                    self.global_router = GlobalRouter(self.channels, self.num_ops)
                    self.spatial_router = SpatialRouter(self.channels, self.num_ops)

            def _compute_weights(self, x: torch.Tensor) -> torch.Tensor:
                if self.cfg.router_mode == "uniform":
                    b, _, h, w = x.shape
                    return torch.full((b, self.num_ops, h, w), 1.0 / self.num_ops, device=x.device, dtype=x.dtype)

                g_logits = self.global_router(x)  # [B,N,1,1]
                s_logits = self.spatial_router(x) # [B,N,H,W]
                logits = (g_logits + s_logits) / max(self.cfg.temperature, self.cfg.eps)
                return torch.softmax(logits, dim=1)

            def _apply_topk(self, weights: torch.Tensor) -> torch.Tensor:
                if self.cfg.router_mode != "dmor" or self.cfg.topk <= 0 or self.cfg.topk >= self.num_ops:
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

    # -------- net.py --------
    (models_dir / "net.py").write_text(dedent(r"""
        from __future__ import annotations
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from .dmor import DMOR


        class ConvBnRelu(nn.Module):
            def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)


        class LiteBlock(nn.Module):
            def __init__(self, in_c: int, out_c: int):
                super().__init__()
                self.in_c = int(in_c)
                self.out_c = int(out_c)
                self.block = nn.Sequential(
                    nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False),
                    nn.BatchNorm2d(in_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                y = self.block(x)
                return x + y if (self.in_c == self.out_c) else y


        class MultiScaleBackbone(nn.Module):
            def __init__(self, base_channels: int = 32):
                super().__init__()
                c = int(base_channels)

                self.stage1_proj = ConvBnRelu(3, c)
                self.stage1_block = LiteBlock(c, c)

                self.stage2_down = ConvBnRelu(c, c * 2, s=2)
                self.stage2_block = LiteBlock(c * 2, c * 2)

                self.stage3_down = ConvBnRelu(c * 2, c * 4, s=2)
                self.stage3_block = LiteBlock(c * 4, c * 4)

            def forward(self, x):
                x1 = self.stage1_block(self.stage1_proj(x))
                x2 = self.stage2_block(self.stage2_down(x1))
                x3 = self.stage3_block(self.stage3_down(x2))
                return x1, x2, x3


        class DMOREdgeNet(nn.Module):
            def __init__(self, channels: int = 32, topk: int = 2, router_mode: str = "dmor",
                         temperature: float = 1.0, backbone: str = "lite",
                         enabled_ops: list[int] | None = None,
                         pool_mode: str = "dmor"):
                super().__init__()
                self.backbone = MultiScaleBackbone(channels)

                self.dmor1 = DMOR(channels, topk=topk, router_mode=router_mode, temperature=temperature,
                                  enabled_ops=enabled_ops, pool_mode=pool_mode)
                self.dmor2 = DMOR(channels * 2, topk=topk, router_mode=router_mode, temperature=temperature,
                                  enabled_ops=enabled_ops, pool_mode=pool_mode)
                self.dmor3 = DMOR(channels * 4, topk=topk, router_mode=router_mode, temperature=temperature,
                                  enabled_ops=enabled_ops, pool_mode=pool_mode)

                self.side1 = nn.Conv2d(channels, 1, 1)
                self.side2 = nn.Conv2d(channels * 2, 1, 1)
                self.side3 = nn.Conv2d(channels * 4, 1, 1)

                self.fuse = nn.Conv2d(3, 1, 1)

            def forward(self, x, return_weights: bool = False):
                f1, f2, f3 = self.backbone(x)
                img_h, img_w = x.shape[2], x.shape[3]

                f1_d = self.dmor1(f1)
                f2_d = self.dmor2(f2)
                f3_d = self.dmor3(f3)

                o1 = self.side1(f1_d)

                o2 = self.side2(f2_d)
                o2 = F.interpolate(o2, size=(img_h, img_w), mode="bilinear", align_corners=False)

                o3 = self.side3(f3_d)
                o3 = F.interpolate(o3, size=(img_h, img_w), mode="bilinear", align_corners=False)

                fused = self.fuse(torch.cat([o1, o2, o3], dim=1))

                if self.training:
                    return [o1, o2, o3, fused]
                return fused
    """).lstrip(), encoding="utf-8")


def run_cmd(cmd: list[str], env: dict, cwd: Path) -> None:
    print("\n[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


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
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ckpt_name", default="dmor_best.pth")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_train", action="store_true")
    ap.add_argument("--no_export", action="store_true")
    ap.add_argument("--mst", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()

    overlay = Path.cwd() / "_overlay_dmor"
    if overlay.exists():
        shutil.rmtree(overlay)
    write_overlay(overlay)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{overlay}:{repo_root}:{env.get('PYTHONPATH','')}"
    print("Operator order:", ", ".join(OPERATOR_ORDER))
    print("Overlay:", overlay)
    print("PYTHONPATH:", env["PYTHONPATH"])

    def run_one(tag: str, enabled_ops: str, pool_mode: str):
        out_dir = outputs_root / tag
        ckpt_dir = out_dir / "ckpt"
        pred_dir = out_dir / "test_png"
        eval_dir = out_dir / "eval_official_gpu"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print("Running:", tag)
        print("enabled_ops:", enabled_ops if enabled_ops else "ALL")
        print("pool_mode:", pool_mode)
        print("out_dir:", out_dir)
        print("==============================")

        # Train: pass ablation controls via env (no script edits needed)
        if not args.no_train:
            env2 = env.copy()
            env2["DMOR_ENABLED_OPS"] = enabled_ops
            env2["DMOR_POOL_MODE"] = pool_mode

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
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
                "--backbone", "lite",
                "--amp",
            ]
            run_cmd(cmd, env=env2, cwd=repo_root)

        # Export
        if not args.no_export:
            ckpt_path = ckpt_dir / args.ckpt_name
            if not ckpt_path.exists():
                raise SystemExit(f"Checkpoint not found: {ckpt_path} (set --ckpt_name)")

            env2 = env.copy()
            env2["DMOR_ENABLED_OPS"] = enabled_ops
            env2["DMOR_POOL_MODE"] = pool_mode

            cmd = [
                "python", "scripts/bsds_export.py",
                "--input_dir", str(data_root / "images/test"),
                "--output_dir", str(pred_dir),
                "--checkpoint", str(ckpt_path),
                "--channels", str(args.channels),
                "--topk", str(args.topk),
                "--router_mode", args.router_mode,
                "--temperature", str(args.temperature),
            ]
            if args.mst:
                cmd.append("--mst")
            run_cmd(cmd, env=env2, cwd=repo_root)

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

    # B1~B5
    run_one(f"{args.exp_prefix}_B1_noO1", "1,2,3,4", "dmor")
    run_one(f"{args.exp_prefix}_B2_noO2", "0,2,3,4", "dmor")
    run_one(f"{args.exp_prefix}_B3_noO3", "0,1,3,4", "dmor")
    run_one(f"{args.exp_prefix}_B4_noO4", "0,1,2,4", "dmor")
    run_one(f"{args.exp_prefix}_B5_noO5", "0,1,2,3", "dmor")

    # B6: Replace All with 3x3
    run_one(f"{args.exp_prefix}_B6_all3x3", "", "all3x3")

    print("\n✅ All ablations finished.")
    print("Outputs root:", outputs_root)


if __name__ == "__main__":
    main()
