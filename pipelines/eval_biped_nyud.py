import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet
from scripts.biped_nyud_train import DMORFusionWrapper, EdgeDataset
from torch.utils.data import DataLoader

def non_maximum_suppression(edge_map):
    """Morphology-based edge thinning (NMS alternative, no cv2.ximgproc dependency)."""
    edge_map = (edge_map * 255.0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(edge_map, kernel, iterations=1)
    edge_map_nms = cv2.absdiff(edge_map, erode)
    edge_map_nms = cv2.GaussianBlur(edge_map, (3, 3), 0)
    return edge_map_nms.astype(np.float32) / 255.0

def calculate_metrics(preds_list, targets_list, name):
    """
    Compute ODS, OIS, and AP for academic edge detection benchmarks.

    Args:
        preds_list: list of flattened per-image prediction arrays
        targets_list: list of flattened per-image ground truth arrays
    """
    print(f"[{name}] Computing metrics (ODS, OIS, AP)...")

    ois_f1_scores = []
    for pred, target in zip(preds_list, targets_list):
        if np.sum(target) == 0:
            continue
        precisions, recalls, thresholds = precision_recall_curve(target, pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        ois_f1_scores.append(np.max(f1_scores))

    ois = np.mean(ois_f1_scores) if len(ois_f1_scores) > 0 else 0.0

    preds_flat = np.concatenate(preds_list)
    targets_flat = np.concatenate(targets_list)

    precisions_global, recalls_global, thresholds_global = precision_recall_curve(targets_flat, preds_flat)
    f1_scores_global = (2 * precisions_global * recalls_global) / (precisions_global + recalls_global + 1e-8)

    ods = np.max(f1_scores_global)
    order = np.argsort(recalls_global)
    recalls_sorted = recalls_global[order]
    precisions_sorted = precisions_global[order]

    ap = np.trapz(precisions_sorted, recalls_sorted)

    print(f"[{name}] ODS: {ods:.4f} | OIS: {ois:.4f} | AP: {ap:.4f}")
    return {
        "ODS": float(ods),
        "OIS": float(ois),
        "AP": float(ap),
        "precision_curve": precisions_sorted.tolist(),
        "recall_curve": recalls_sorted.tolist(),
    }

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_nyud = (args.dataset == 'NYUDv2')

    max_dist = 0.011 if is_nyud else 0.015
    print(f"Evaluating {args.dataset} | maxDist: {max_dist}")

    img_size = (480, 640) if is_nyud else (720, 1280)
    test_dataset = EdgeDataset(args.data_root, args.dataset, is_train=False, img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if is_nyud:
        model = DMORFusionWrapper(channels=args.channels).to(device)
    else:
        model = DMOREdgeNet(channels=args.channels).to(device)

    print(f"Loading weights from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_targets = []
    if is_nyud:
        preds_rgb, preds_hha, preds_fusion = [], [], []
    else:
        preds_biped = []

    print("Running inference on test set...")
    with torch.no_grad():
        for data in test_loader:
            if is_nyud:
                img_rgb, img_hha, target = data
                out_rgb, out_hha, out_fusion = model(img_rgb.to(device), img_hha.to(device))

                preds_rgb.append(non_maximum_suppression(torch.sigmoid(out_rgb).cpu().numpy().squeeze()).flatten())
                preds_hha.append(non_maximum_suppression(torch.sigmoid(out_hha).cpu().numpy().squeeze()).flatten())
                preds_fusion.append(non_maximum_suppression(torch.sigmoid(out_fusion).cpu().numpy().squeeze()).flatten())
            else:
                img_rgb, target = data
                out_biped = model(img_rgb.to(device))
                preds_biped.append(non_maximum_suppression(torch.sigmoid(out_biped).cpu().numpy().squeeze()).flatten())

            all_targets.append(target.numpy().squeeze().flatten())

    print("\n--- Results ---")
    if is_nyud:
        res_rgb = calculate_metrics(preds_rgb, all_targets, "NYUD - RGB Only")
        res_hha = calculate_metrics(preds_hha, all_targets, "NYUD - HHA Only")
        res_fusion = calculate_metrics(preds_fusion, all_targets, "NYUD - RGB-HHA Fusion")

        import json
        save_dir = args.save_dir if args.save_dir else "eval_results"
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "NYUD_RGB.json"), "w") as f:
            json.dump(res_rgb, f, indent=2)
        with open(os.path.join(save_dir, "NYUD_HHA.json"), "w") as f:
            json.dump(res_hha, f, indent=2)
        with open(os.path.join(save_dir, "NYUD_FUSION.json"), "w") as f:
            json.dump(res_fusion, f, indent=2)

        print(f"Saved NYUD metrics to {save_dir}")
    else:
        res = calculate_metrics(preds_biped, all_targets, "BIPED - RGB")

        import json
        save_dir = args.save_dir if args.save_dir else "eval_results"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['BIPED', 'NYUDv2'])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="eval_results")
    evaluate(parser.parse_args())
