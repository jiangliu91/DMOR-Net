import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet

def multi_scale_inference(model, img, device, scales=[0.5, 1.0, 1.5]):
    """Multi-scale test-time augmentation (MS-Test) with horizontal flip."""
    h, w = img.shape[:2]
    all_preds = []

    for scale in scales:
        img_s = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_s.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        out_s = torch.sigmoid(model(img_t)).cpu().numpy().squeeze()
        out_s = cv2.resize(out_s, (w, h), interpolation=cv2.INTER_LINEAR)
        all_preds.append(out_s)

        img_flip = img_s[:, ::-1, :]
        img_flip_t = torch.from_numpy(img_flip.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        out_flip = torch.sigmoid(model(img_flip_t)).cpu().numpy().squeeze()
        out_flip = cv2.resize(out_flip[:, ::-1], (w, h), interpolation=cv2.INTER_LINEAR)
        all_preds.append(out_flip)

    return np.mean(all_preds, axis=0)

def save_predictions(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running multi-scale inference (MS-Test) export...")

    test_img_dir = Path(args.data_root) / 'images' / 'test'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DMOREdgeNet(channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img_paths = sorted([p for p in test_img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')])

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Generating edge maps"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = multi_scale_inference(model, img, device)
            pred_uint8 = (pred * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), pred_uint8)

    print(f"Predictions saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    save_predictions(parser.parse_args())
