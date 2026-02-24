import torch
import numpy as np
import cv2
import argparse
from sklearn.metrics import precision_recall_curve
from model import DMOREdge

def non_maximum_suppression(edge_map):
    edge_map_nms = cv2.ximgproc.createEdgeAwareFilters().dtFilter(
        edge_map, edge_map, sigmaSpatial=1.0, sigmaColor=1.0)
    return edge_map_nms

def calculate_metrics(preds_flat, targets_flat, name):
    precisions, recalls, thresholds = precision_recall_curve(targets_flat, preds_flat)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    ods = np.max(f1_scores)
    ap = np.trapz(precisions, recalls)
    print(f"[{name}] ODS: {ods:.4f} | AP: {ap:.4f}")

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_nyud = (args.dataset == 'NYUDv2')
    
    # 严格对齐评估基准配置 
    max_dist = 0.011 if is_nyud else 0.015 
    print(f"Evaluating {args.dataset} with maxDist: {max_dist}")

    model = DMOREdge(in_channels_rgb=3, in_channels_hha=3 if is_nyud else 0).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    test_loader = create_dataloader(args.dataset, batch_size=1, is_train=False)
    
    if is_nyud:
        preds_rgb, preds_hha, preds_fusion, all_targets =,,,
    else:
        preds_biped, all_targets =,

    with torch.no_grad():
        for data in test_loader:
            if is_nyud:
                img_rgb, img_hha, target = data
                out_rgb, out_hha, out_fusion = model(img_rgb.to(device), img_hha.to(device))
                
                preds_rgb.append(non_maximum_suppression(torch.sigmoid(out_rgb[-1]).cpu().numpy().squeeze()).flatten())
                preds_hha.append(non_maximum_suppression(torch.sigmoid(out_hha[-1]).cpu().numpy().squeeze()).flatten())
                preds_fusion.append(non_maximum_suppression(torch.sigmoid(out_fusion[-1]).cpu().numpy().squeeze()).flatten())
            else:
                img_rgb, target = data
                out_biped = model(img_rgb.to(device), None)
                preds_biped.append(non_maximum_suppression(torch.sigmoid(out_biped[-1]).cpu().numpy().squeeze()).flatten())
            
            all_targets.append(target.numpy().squeeze().flatten())

    targets_flat = np.concatenate(all_targets)
    if is_nyud:
        print("--- NYUDv2 Results ---")
        calculate_metrics(np.concatenate(preds_rgb), targets_flat, "RGB Only")
        calculate_metrics(np.concatenate(preds_hha), targets_flat, "HHA Only")
        calculate_metrics(np.concatenate(preds_fusion), targets_flat, "RGB-HHA Fusion")
    else:
        print("--- BIPED Results ---")
        calculate_metrics(np.concatenate(preds_biped), targets_flat, "RGB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=)
    parser.add_argument("--checkpoint", type=str, required=True)
    evaluate(parser.parse_args())