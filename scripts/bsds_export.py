# scripts/bsds_export.py
# BSDS500 test export -> PNG (for official BSDS evaluation)
# This version is CONDA-friendly and supports env vars so you don't have to keep editing the file.

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from models.net import DMOREdgeNet


def _env(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def load_image_rgb(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


@torch.no_grad()
def main():
    # You can override these via environment variables:
    #   BSDS_ROOT, OUT_DIR, CKPT_PATH, BACKBONE, TOPK, ROUTER_MODE, TEMPERATURE
    bsds_root = _env("BSDS_ROOT", r"D:\Users\JJzhe\code\github\dataset\BSDS500")
    out_dir   = _env("OUT_DIR",   r"D:\Users\JJzhe\code\github\outputs\BSDS500\DMOR\test_png")
    ckpt_path = _env("CKPT_PATH", "")  # optional

    backbone  = _env("BACKBONE", "lite")
    topk      = int(_env("TOPK", "2"))
    router    = _env("ROUTER_MODE", "dmor")
    temp      = float(_env("TEMPERATURE", "1.0"))

    img_dir = os.path.join(bsds_root, "images", "test")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DMOREdgeNet(channels=32, topk=topk, router_mode=router, temperature=temp, backbone=backbone).to(device)

    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.eval()

    exts = {".jpg", ".png", ".jpeg"}
    img_paths = [p for p in sorted(Path(img_dir).iterdir()) if p.suffix.lower() in exts]
    if not img_paths:
        raise SystemExit(f"[ERROR] no images under: {img_dir}")

    print(f"[INFO] device={device} | backbone={backbone} topk={topk} router={router} temp={temp}")
    print(f"[INFO] exporting {len(img_paths)} images -> {out_dir}")

    for p in img_paths:
        x = load_image_rgb(str(p)).to(device)
        logits = model(x)  # [1,1,H,W]
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        pred = (prob * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, p.stem + ".png"), pred)

    print("✅ export done")


if __name__ == "__main__":
    main()
