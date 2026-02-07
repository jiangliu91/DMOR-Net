
import os
import json
import argparse
import numpy as np
import cv2
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# -----------------------------------------------------------------------------
# HARD REQUIREMENT: GPU + CUDA only
# -----------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available. Please run with a CUDA-enabled PyTorch.")

# CuPy for GPU EDT (distance transform)
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpx_ndi
except Exception as e:
    raise RuntimeError(
        "CuPy (CUDA build) is required for GPU EDT. Install a matching wheel, e.g. cupy-cuda12x.\n"
        f"Original import error: {repr(e)}"
    )

# -----------------------------------------------------------------------------
# GPU Zhang-Suen thinning (vectorized) in PyTorch
# -----------------------------------------------------------------------------
@torch.no_grad()
def zhang_suen_thinning(x: torch.Tensor, max_iter: int = 80) -> torch.Tensor:
    """
    Zhang-Suen thinning on GPU.
    Args:
        x: (H, W) bool/uint8 tensor on CUDA, where 1 indicates foreground (edge).
    Returns:
        thinned x: (H, W) bool tensor on CUDA.
    Notes:
        - This is a standard two-subiteration Zhang-Suen algorithm, fully vectorized.
        - Iterates until convergence or max_iter.
    """
    if x.dtype != torch.bool:
        x = x > 0
    x = x.clone()

    # helper: roll-based neighbors (p2..p9) around each pixel p1
    def neighbors(img: torch.Tensor):
        p2 = torch.roll(img, shifts=-1, dims=0)      # up
        p3 = torch.roll(torch.roll(img, -1, 0),  1, 1)  # up-right
        p4 = torch.roll(img, shifts=1, dims=1)       # right
        p5 = torch.roll(torch.roll(img,  1, 0),  1, 1)  # down-right
        p6 = torch.roll(img, shifts=1, dims=0)       # down
        p7 = torch.roll(torch.roll(img,  1, 0), -1, 1)  # down-left
        p8 = torch.roll(img, shifts=-1, dims=1)      # left
        p9 = torch.roll(torch.roll(img, -1, 0), -1, 1)  # up-left
        return p2, p3, p4, p5, p6, p7, p8, p9

    for _ in range(max_iter):
        prev = x

        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(x)
        # number of non-zero neighbors
        B = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).to(torch.uint8)

        # number of 0->1 transitions in ordered sequence p2,p3,...,p9,p2
        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        A = torch.zeros_like(B, dtype=torch.uint8)
        for i in range(8):
            A += ((~seq[i]) & seq[i + 1]).to(torch.uint8)

        # Common conditions
        cond0 = x
        cond1 = (B >= 2) & (B <= 6)
        cond2 = (A == 1)

        # Sub-iteration 1 conditions
        cond3_1 = (~(p2 & p4 & p6))
        cond4_1 = (~(p4 & p6 & p8))
        m1 = cond0 & cond1 & cond2 & cond3_1 & cond4_1

        x1 = x & (~m1)

        # Sub-iteration 2
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(x1)
        B = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).to(torch.uint8)

        seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        A = torch.zeros_like(B, dtype=torch.uint8)
        for i in range(8):
            A += ((~seq[i]) & seq[i + 1]).to(torch.uint8)

        cond0 = x1
        cond1 = (B >= 2) & (B <= 6)
        cond2 = (A == 1)

        cond3_2 = (~(p2 & p4 & p8))
        cond4_2 = (~(p2 & p6 & p8))
        m2 = cond0 & cond1 & cond2 & cond3_2 & cond4_2

        x2 = x1 & (~m2)

        # Check convergence
        if torch.equal(x2, prev):
            x = x2
            break
        x = x2

    return x

# -----------------------------------------------------------------------------
# GPU EDT via CuPy (no CPU copy) using DLPack
# -----------------------------------------------------------------------------
def edt_gpu_bool_to_float(inv_bool_torch: torch.Tensor) -> "cp.ndarray":
    """
    Compute Euclidean distance transform on GPU using CuPy.
    Args:
        inv_bool_torch: (H, W) bool/uint8 torch CUDA tensor
                        True(1) indicates background for EDT, False(0) indicates sources.
                        This matches distance_transform_edt where distance to nearest False.
    Returns:
        dist: cupy.ndarray float32 on GPU
    """
    if inv_bool_torch.dtype != torch.uint8:
        inv_bool_torch = inv_bool_torch.to(torch.uint8)
    inv_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(inv_bool_torch))
    dist_cp = cpx_ndi.distance_transform_edt(inv_cp)  # float64 by default
    return dist_cp.astype(cp.float32)

# -----------------------------------------------------------------------------
# Matching for a single GT using symmetric DT (pred->gt for precision, gt->pred for recall)
# -----------------------------------------------------------------------------
def match_counts(pred_bin: torch.Tensor, gt_bin: torch.Tensor, max_dist: float, thin: bool) -> tuple[float, float, float, float]:
    """
    Args:
        pred_bin: (H, W) bool CUDA
        gt_bin:   (H, W) bool CUDA
        max_dist: float (pixels)
        thin: whether to thin pred & gt before matching
    Returns:
        tp_p, n_p, tp_r, n_r (floats)
    """
    if thin:
        pred = zhang_suen_thinning(pred_bin)
        gt = zhang_suen_thinning(gt_bin)
    else:
        pred = pred_bin
        gt = gt_bin

    n_p = float(pred.sum().item())
    n_r = float(gt.sum().item())

    if n_p == 0.0:
        return 0.0, 0.0, 0.0, n_r
    if n_r == 0.0:
        return 0.0, n_p, 0.0, 0.0

    # DT of GT for precision (distance from every pixel to nearest GT edge)
    inv_gt = (~gt).to(torch.uint8)  # True/1 background, False/0 sources
    dist_to_gt = edt_gpu_bool_to_float(inv_gt)  # cupy float32

    # DT of PRED for recall
    inv_pred = (~pred).to(torch.uint8)
    dist_to_pred = edt_gpu_bool_to_float(inv_pred)  # cupy float32

    # Convert pred/gt mask to cupy (no copy)
    pred_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(pred.to(torch.uint8)))
    gt_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(gt.to(torch.uint8)))

    # TP for precision: pred pixels within max_dist of GT
    tp_p = float(cp.count_nonzero((pred_cp != 0) & (dist_to_gt <= max_dist)))

    # TP for recall: GT pixels within max_dist of pred
    tp_r = float(cp.count_nonzero((gt_cp != 0) & (dist_to_pred <= max_dist)))

    return tp_p, n_p, tp_r, n_r

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class BSDSDataset(Dataset):
    def __init__(self, img_list, gt_dir, pred_dir):
        self.img_list = img_list
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        stem = img_path.stem

        pred_path = self.pred_dir / f"{stem}.png"
        if not pred_path.exists():
            return None

        pred_cv = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_cv is None:
            return None
        pred = pred_cv.astype(np.float32) / 255.0

        gt_path = self.gt_dir / f"{stem}.mat"
        if not gt_path.exists():
            return None
        mat = scipy.io.loadmat(str(gt_path))
        gt_struct = mat["groundTruth"]

        gts = []
        for i in range(gt_struct.size):
            gt_entry = gt_struct[0, i]
            try:
                b_map = gt_entry["Boundaries"][0, 0]
            except Exception:
                continue
            if b_map.ndim == 2:
                gts.append(b_map.astype(np.bool_))

        if len(gts) == 0:
            return None

        return {"stem": stem, "pred": pred, "gts": gts, "shape": pred.shape}

def collate_fn_bsds(batch):
    batch = [b for b in batch if b is not None]
    return batch

# -----------------------------------------------------------------------------
# AP with precision envelope (BSDS/VOC style)
# -----------------------------------------------------------------------------
def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    if rec.size == 0:
        return 0.0
    order = np.argsort(rec)
    rec = rec[order]
    prec = prec[order]

    # precision envelope
    for i in range(prec.size - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # integrate
    return float(np.trapz(prec, rec))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsds_root", type=str, required=True, help="Path to BSDS500 root")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to predictions (.png)")
    parser.add_argument("--out_dir", type=str, default="./results_gpu_official")
    parser.add_argument("--thresholds", type=int, default=99, help="Number of thresholds")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--thin", action="store_true", help="Apply GPU thinning (recommended for BSDS official style)")
    args = parser.parse_args()

    device = torch.device("cuda")
    bsds_root = Path(args.bsds_root)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = bsds_root / "groundTruth/test"
    if not gt_dir.exists():
        gt_dir = bsds_root / "data/groundTruth/test"

    img_dir = bsds_root / "images/test"
    if not img_dir.exists():
        img_dir = bsds_root / "data/images/test"

    img_list = sorted(list(img_dir.glob("*.jpg")))
    if len(img_list) == 0:
        raise RuntimeError(f"No test images found under: {img_dir}")

    dataset = BSDSDataset(img_list, gt_dir, pred_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn_bsds)

    # thresholds: avoid the pathological all-ones map at thr==0
    # use [1/255, 1] by default for 8-bit predictions
    thresh_vals = torch.linspace(1.0 / 255.0, 1.0, args.thresholds, device=device)

    total_tp = torch.zeros(args.thresholds, device=device, dtype=torch.float64)
    total_np = torch.zeros(args.thresholds, device=device, dtype=torch.float64)
    total_tr = torch.zeros(args.thresholds, device=device, dtype=torch.float64)
    total_nr = torch.zeros(args.thresholds, device=device, dtype=torch.float64)

    ois_f = []

    print(f"Start Eval: {len(img_list)} images | GPU: {torch.cuda.get_device_name(0)} | thin={args.thin}")

    for batch in tqdm(loader):
        if not batch:
            continue
        data = batch[0]

        pred_raw = torch.from_numpy(data["pred"]).to(device)  # (H,W) float32
        gts = [torch.from_numpy(gt).to(device) for gt in data["gts"]]  # list of bool on cuda

        H, W = pred_raw.shape
        max_dist = float(0.0075 * np.sqrt(H * H + W * W))  # pixels

        img_tp = torch.zeros(args.thresholds, device=device, dtype=torch.float64)
        img_np = torch.zeros(args.thresholds, device=device, dtype=torch.float64)
        img_tr = torch.zeros(args.thresholds, device=device, dtype=torch.float64)
        img_nr = torch.zeros(args.thresholds, device=device, dtype=torch.float64)

        # For OIS: best F over thresholds (after best-GT selection per threshold)
        best_f_for_img = 0.0

        for t_idx, thr in enumerate(thresh_vals):
            pred_bin = (pred_raw >= thr)

            # Evaluate against each GT; pick the GT that yields best F at this threshold (standard practice)
            best = None
            best_f = -1.0

            for gt in gts:
                tp_p, n_p, tp_r, n_r = match_counts(pred_bin, gt.bool(), max_dist=max_dist, thin=args.thin)

                # Convert to float for metric
                P = tp_p / (n_p + 1e-8) if n_p > 0 else 0.0
                R = tp_r / (n_r + 1e-8) if n_r > 0 else 0.0
                F = (2 * P * R) / (P + R + 1e-8) if (P + R) > 0 else 0.0

                if F > best_f:
                    best_f = F
                    best = (tp_p, n_p, tp_r, n_r)

            if best is None:
                continue

            tp_p, n_p, tp_r, n_r = best
            img_tp[t_idx] = tp_p
            img_np[t_idx] = n_p
            img_tr[t_idx] = tp_r
            img_nr[t_idx] = n_r

            if best_f > best_f_for_img:
                best_f_for_img = best_f

        total_tp += img_tp
        total_np += img_np
        total_tr += img_tr
        total_nr += img_nr

        ois_f.append(best_f_for_img)

    ods_P = total_tp / (total_np + 1e-8)
    ods_R = total_tr / (total_nr + 1e-8)
    ods_F = (2 * ods_P * ods_R) / (ods_P + ods_R + 1e-8)

    best_ods_idx = torch.argmax(ods_F).item()
    ODS = float(ods_F[best_ods_idx].item())
    OIS = float(np.mean(ois_f)) if len(ois_f) > 0 else 0.0

    rec_np = ods_R.detach().cpu().numpy()
    prec_np = ods_P.detach().cpu().numpy()
    AP = average_precision(rec_np, prec_np)

    results = {
        "ODS": ODS,
        "OIS": OIS,
        "AP": AP,
        "ODS_Threshold": float(thresh_vals[best_ods_idx].item()),
        "thin": bool(args.thin),
        "thresholds": int(args.thresholds),
        "max_dist_rule": "0.0075 * diag(H,W)",
        "dt": "cupy distance_transform_edt (GPU)",
        "thinning": "Zhang-Suen (GPU, torch)",
    }

    print("\n" + "=" * 32)
    print("   BSDS500 EVAL (GPU, fixed)  ")
    print("=" * 32)
    print(json.dumps(results, indent=2))

    with open(out_dir / "eval_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
