import os
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------
# FIX: ensure project root in PYTHONPATH
# ------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.net import DMOREdgeNet


def _read_rgb_float01(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0  # match bsds_train.py


def _to_tensor(img_rgb01: np.ndarray, device: torch.device) -> torch.Tensor:
    # [H,W,3] -> [1,3,H,W]
    return torch.from_numpy(img_rgb01).permute(2, 0, 1).unsqueeze(0).to(device)


def _minmax_uint8(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # contrast stretch for visibility/quality; does NOT change ordering much
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return (x * 255.0).clip(0, 255).astype(np.uint8)
    y = (x - lo) / (hi - lo)
    return (y * 255.0).clip(0, 255).astype(np.uint8)


def infer_one(model, img_path: str, device: torch.device, scales=(1.0,), flip_tta: bool = True) -> np.ndarray:
    img = _read_rgb_float01(img_path)
    H, W, _ = img.shape

    acc = torch.zeros((1, 1, H, W), device=device)
    cnt = 0

    for s in scales:
        hs, ws = max(1, int(H * s)), max(1, int(W * s))
        img_s = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)

        x = _to_tensor(img_s, device)

        with torch.no_grad():
            logits = model(x)             # expects logits
            prob = torch.sigmoid(logits)  # [1,1,hs,ws]
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
        acc += prob
        cnt += 1

        if flip_tta:
            x_f = torch.flip(x, dims=[3])
            with torch.no_grad():
                logits_f = model(x_f)
                prob_f = torch.sigmoid(logits_f)
            prob_f = torch.flip(prob_f, dims=[3])
            prob_f = F.interpolate(prob_f, size=(H, W), mode="bilinear", align_corners=False)
            acc += prob_f
            cnt += 1

    prob = (acc / max(1, cnt)).squeeze().clamp(0, 1).cpu().numpy()  # [H,W]
    return prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--router_mode", type=str, default="dmor", choices=["dmor", "uniform"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--mst", action="store_true", help="multi-scale + flip tta")
    p.add_argument("--no_flip", action="store_true")
    p.add_argument("--stretch", action="store_true", help="min-max stretch before saving (often improves visibility)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DMOREdgeNet(
        channels=args.channels,
        topk=args.topk,
        router_mode=args.router_mode,
        temperature=args.temperature,
        backbone="lite",
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    imgs = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    print(f"Found {len(imgs)} images. MST={args.mst} flip={not args.no_flip}")

    scales = (0.5, 1.0, 1.5) if args.mst else (1.0,)

    for i, name in enumerate(imgs, 1):
        prob = infer_one(model, os.path.join(args.input_dir, name), device, scales=scales, flip_tta=not args.no_flip)
        if args.stretch:
            out = _minmax_uint8(prob)
        else:
            out = (prob * 255.0).clip(0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.output_dir, os.path.splitext(name)[0] + ".png"), out)

        if i % 10 == 0 or i == len(imgs):
            print(f"Processed {i}/{len(imgs)}")

    print("Export done.")


if __name__ == "__main__":
    main()