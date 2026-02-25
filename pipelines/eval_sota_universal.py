
import os
import sys
import time
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

try:
    from thop import profile
    THOP_AVAILABLE = True
except:
    THOP_AVAILABLE = False

# Add project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet


# ============================
# Model Complexity
# ============================

def compute_complexity(model, device, input_size=(1,3,512,512)):
    print("\n====== Model Complexity ======")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params/1e6:.3f} M")

    if THOP_AVAILABLE:
        dummy = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs:  {flops/1e9:.3f} G")
    else:
        print("FLOPs:  thop not installed")

    # Pure forward FPS
    dummy = torch.randn(*input_size).to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(20):  # warmup
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        iters = 100
        for _ in range(iters):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    pure_fps = iters / (end - start)
    print(f"FPS (pure forward): {pure_fps:.2f}")
    print("================================\n")


# ============================
# Real Inference FPS
# ============================

def compute_real_fps(model, args, device):
    model.eval()
    img_dir = Path(args.data_root) / "images" / "test"
    img_paths = sorted(img_dir.glob("*.*"))

    start = time.time()

    with torch.no_grad():
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_t = torch.from_numpy(
                img.astype(np.float32)/255.0
            ).permute(2,0,1).unsqueeze(0).to(device)

            out = model(img_t)
            if isinstance(out, list):
                out = out[-1]

            pred = torch.sigmoid(out)
            _ = pred.cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    real_fps = len(img_paths) / (end - start)
    print(f"FPS (real inference): {real_fps:.2f}\n")


# ============================
# Evaluation
# ============================

def compute_matches_with_tolerance(pred_b, gt_b, max_dist):
    if not np.any(pred_b) or not np.any(gt_b):
        return 0, np.sum(pred_b), np.sum(gt_b)

    dist_gt = distance_transform_edt(~gt_b)
    dist_pred = distance_transform_edt(~pred_b)

    match_pred = (dist_gt <= max_dist) & pred_b
    tp = np.sum(match_pred)
    fp = np.sum(pred_b) - tp

    match_gt = (dist_pred <= max_dist) & gt_b
    fn = np.sum(gt_b) - np.sum(match_gt)

    return tp, fp, fn


def evaluate(preds, gts, max_dist_ratio=0.015):
    thresholds = np.linspace(0.01, 0.99, 99)
    sum_tp = np.zeros(99)
    sum_fp = np.zeros(99)
    sum_fn = np.zeros(99)
    all_ois_f1 = []

    print("Evaluating...")

    for i in tqdm(range(len(preds))):
        pred = preds[i]
        gt = (gts[i] > 0.5)

        diag = np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)
        pixel_tolerance = max_dist_ratio * diag

        img_best_f1 = 0

        for t_idx, t in enumerate(thresholds):
            pred_b = (pred >= t)
            tp, fp, fn = compute_matches_with_tolerance(pred_b, gt, pixel_tolerance)

            sum_tp[t_idx] += tp
            sum_fp[t_idx] += fp
            sum_fn[t_idx] += fn

            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            img_best_f1 = max(img_best_f1, f1)

        all_ois_f1.append(img_best_f1)

    precisions = sum_tp / (sum_tp + sum_fp + 1e-8)
    recalls = sum_tp / (sum_tp + sum_fn + 1e-8)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx = np.argmax(f1_scores)
    ods = f1_scores[best_idx]
    ois = np.mean(all_ois_f1)
    ap = np.abs(np.trapz(precisions[::-1], recalls[::-1]))

    print("\n====== Final Evaluation ======")
    print(f"ODS: {ods:.4f}")
    print(f"OIS: {ois:.4f}")
    print(f"AP:  {ap:.4f}")
    print(f"Precision@ODS: {precisions[best_idx]:.4f}")
    print(f"Recall@ODS:    {recalls[best_idx]:.4f}")
    print("================================\n")


# ============================
# Main
# ============================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DMOREdgeNet(channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    compute_complexity(model, device)
    compute_real_fps(model, args, device)

    preds = []
    gts = []

    img_dir = Path(args.data_root) / "images" / "test"
    gt_dir = Path(args.data_root) / "gt" / "test"

    with torch.no_grad():
        for img_path in tqdm(sorted(img_dir.glob("*.*"))):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_t = torch.from_numpy(
                img.astype(np.float32)/255.0
            ).permute(2,0,1).unsqueeze(0).to(device)

            out = model(img_t)
            if isinstance(out, list):
                out = out[-1]

            pred = torch.sigmoid(out).cpu().numpy().squeeze()

            gt_path = gt_dir / f"{img_path.stem}.png"
            gt = cv2.imread(str(gt_path), 0)
            if gt is None:
                continue

            preds.append(pred)
            gts.append(gt.astype(np.float32)/255.0)

    evaluate(preds, gts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    main(parser.parse_args())
