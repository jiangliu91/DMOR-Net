import os
import cv2
import torch
import numpy as np
import argparse
from glob import glob
from scipy.io import loadmat


def _load_gt_stack(gt_path: str):
    """Load BSDS groundTruth .mat -> tensor [N,H,W] float32 {0,1}."""
    mat = loadmat(gt_path)
    gt = mat["groundTruth"]
    n = gt.shape[1] if gt.shape[0] == 1 else gt.shape[0]
    gts = []
    for i in range(n):
        item = gt[0, i] if gt.shape[0] == 1 else gt[i, 0]
        bnd = item["Boundaries"][0, 0].astype(np.float32)
        gts.append(torch.from_numpy((bnd > 0).astype(np.float32)))
    return torch.stack(gts, dim=0)  # [N,H,W]


def _ap_from_pr(P: torch.Tensor, R: torch.Tensor) -> float:
    """Compute AP using precision envelope (VOC-style) on P-R curve."""
    # Sort by recall
    order = torch.argsort(R)
    R = R[order]
    P = P[order]

    # Precision envelope (monotone decreasing)
    P_env = P.clone()
    for i in range(P_env.numel() - 2, -1, -1):
        P_env[i] = torch.maximum(P_env[i], P_env[i + 1])

    # Integrate over recall
    dR = R[1:] - R[:-1]
    ap = torch.sum(dR * P_env[1:]).item()
    return float(ap)


def eval_one_image_cuda(pred_png: str, gt_mat: str, thresholds: torch.Tensor, max_dist=0.0075, device="cuda"):
    """Return per-threshold (tp_p, total_p, tp_r, gt_total) for one image."""
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    gts = _load_gt_stack(gt_mat).to(dev)  # [A,H,W]
    pred = cv2.imread(pred_png, 0)
    if pred is None:
        raise FileNotFoundError(pred_png)
    pred = torch.from_numpy(pred.astype(np.float32) / 255.0).to(dev)  # [H,W]

    H, W = pred.shape
    diagonal = float(np.sqrt(H * H + W * W))
    radius = max_dist * diagonal
    k = int(2 * radius + 1)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    pad = k // 2

    kernel = torch.ones((1, 1, k, k), device=dev)

    # GT match zone (union across annotators)
    match_gt = torch.nn.functional.conv2d(gts.unsqueeze(1), kernel, padding=pad)
    match_gt = (match_gt > 0).float().squeeze(1)          # [A,H,W]
    match_zone = match_gt.max(dim=0).values                # [H,W]

    # total GT positives (sum across annotators, like earlier lightweight eval)
    gt_total = gts.sum(dim=(1, 2)).clamp_min(1.0).sum()    # scalar

    tp_p = torch.zeros_like(thresholds, device=dev)
    total_p = torch.zeros_like(thresholds, device=dev)
    tp_r = torch.zeros_like(thresholds, device=dev)

    for i, t in enumerate(thresholds):
        p_bin = (pred >= t).float()

        # precision numerator/denom (pred pixels matched to GT zone)
        tp_p[i] = (p_bin * match_zone).sum()
        total_p[i] = p_bin.sum().clamp_min(1.0)

        # recall numerator: GT pixels matched to dilated prediction
        p_d = torch.nn.functional.conv2d(p_bin.unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        p_d = (p_d > 0).float().squeeze(0).squeeze(0)
        tp_r[i] = (gts * p_d).sum(dim=(1, 2)).sum()

    return tp_p, total_p, tp_r, gt_total


def run_eval(pred_dir: str, gt_dir: str, device: str = "cuda", threshold_steps: int = 99):
    preds = sorted(glob(os.path.join(pred_dir, "*.png")))
    if len(preds) == 0:
        print("No prediction images found.")
        return

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    thresholds = torch.linspace(1.0 / threshold_steps, 1.0 - 1.0 / threshold_steps, threshold_steps, device=dev)

    # Dataset-level accumulators (for ODS + AP)
    total_tp_p = torch.zeros(threshold_steps, device=dev)
    total_total_p = torch.zeros(threshold_steps, device=dev)
    total_tp_r = torch.zeros(threshold_steps, device=dev)
    total_gt_total = torch.tensor(0.0, device=dev)

    # OIS: average of per-image best F
    ois_sum = 0.0
    ois_count = 0

    print(f"Evaluating {len(preds)} images on {dev} ...")

    for p_path in preds:
        stem = os.path.splitext(os.path.basename(p_path))[0]
        gt_path = os.path.join(gt_dir, f"{stem}.mat")
        if not os.path.exists(gt_path):
            continue

        tp_p, total_p, tp_r, gt_total = eval_one_image_cuda(
            p_path, gt_path, thresholds=thresholds, device=str(dev)
        )

        # Dataset-level sum
        total_tp_p += tp_p
        total_total_p += total_p
        total_tp_r += tp_r
        total_gt_total += gt_total

        # Per-image best F for OIS (lightweight)
        P_i = tp_p / (total_p + 1e-8)
        R_i = tp_r / (gt_total + 1e-8)
        F_i = 2 * P_i * R_i / (P_i + R_i + 1e-8)
        best_f = float(torch.max(F_i).item())
        ois_sum += best_f
        ois_count += 1

    # ODS + AP from dataset PR curve
    P = total_tp_p / (total_total_p + 1e-8)
    R = total_tp_r / (total_gt_total + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)

    ods_idx = int(torch.argmax(F1).item())
    ods = float(F1[ods_idx].item())
    best_t = float((ods_idx + 1) / 100.0)

    ois = (ois_sum / max(1, ois_count)) if ois_count > 0 else 0.0
    ap = _ap_from_pr(P.detach().cpu(), R.detach().cpu())

    print("------------ Evaluation Result ------------")
    print(f"ODS: {ods:.4f} at threshold {best_t:.2f}")
    print(f"OIS: {ois:.4f}")
    print(f"AP : {ap:.4f}")
    print(f"Precision: {float(P[ods_idx].item()):.4f}, Recall: {float(R[ods_idx].item()):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threshold_steps", type=int, default=99)
    args = ap.parse_args()

    with torch.no_grad():
        run_eval(args.pred_dir, args.gt_dir, device=args.device, threshold_steps=args.threshold_steps)