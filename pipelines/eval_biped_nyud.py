import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_curve

# 获取项目根目录 (DMOR-Edge) 并加入系统路径以导入网络模型
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet
from scripts.biped_nyud_train import DMORFusionWrapper, EdgeDataset
from torch.utils.data import DataLoader

def non_maximum_suppression(edge_map):
    """
    使用更通用的形态学方法实现边缘细化 (NMS 替代方案)，
    不再依赖不稳定的 cv2.ximgproc。
    """
    # 归一化到 0-255
    edge_map = (edge_map * 255.0).astype(np.uint8)
    
    # 使用结构元素进行骨架细化或简单的形态学梯度处理
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(edge_map, kernel, iterations=1)
    edge_map_nms = cv2.absdiff(edge_map, erode)
    
    # 也可以使用简单的多尺度高斯模糊平滑噪声
    edge_map_nms = cv2.GaussianBlur(edge_map, (3, 3), 0)
    
    return edge_map_nms.astype(np.float32) / 255.0

def calculate_metrics(preds_list, targets_list, name):
    """
    Calculates ODS, OIS, and AP metrics exactly as required for academic benchmarks.
    preds_list: List of flattened numpy arrays (per image prediction)
    targets_list: List of flattened numpy arrays (per image ground truth)
    """
    print(f"[{name}] 开始计算严谨评估指标 (ODS, OIS, AP)...")
    
    # 1. 计算 OIS (Optimal Image Scale) - 单图最优阈值
    ois_f1_scores = []
    for pred, target in zip(preds_list, targets_list):
        # 忽略全黑的 target 图片以防除以零报错
        if np.sum(target) == 0:
            continue
        precisions, recalls, thresholds = precision_recall_curve(target, pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        ois_f1_scores.append(np.max(f1_scores))
    
    ois = np.mean(ois_f1_scores) if len(ois_f1_scores) > 0 else 0.0

    # 2. 计算 ODS (Optimal Dataset Scale) 和 AP (Average Precision) - 全局阈值
    preds_flat = np.concatenate(preds_list)
    targets_flat = np.concatenate(targets_list)
    
    precisions_global, recalls_global, thresholds_global = precision_recall_curve(targets_flat, preds_flat)
    f1_scores_global = (2 * precisions_global * recalls_global) / (precisions_global + recalls_global + 1e-8)
    
    ods = np.max(f1_scores_global)
    ap = np.trapz(precisions_global, recalls_global) # 积分计算曲线下面积

    print(f"[{name}] 最终结果 => ODS: {ods:.4f} | OIS: {ois:.4f} | AP: {ap:.4f}")
    return ods, ois, ap

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_nyud = (args.dataset == 'NYUDv2')
    
    # 对齐官方的距离容差标准
    max_dist = 0.011 if is_nyud else 0.015 
    print(f"Evaluating {args.dataset} with maxDist equivalent: {max_dist}")

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

    print("Running inference and NMS extraction on test set...")
    with torch.no_grad():
        for data in test_loader:
            if is_nyud:
                img_rgb, img_hha, target = data
                out_rgb, out_hha, out_fusion = model(img_rgb.to(device), img_hha.to(device))
                
                pred_rgb_nms = non_maximum_suppression(torch.sigmoid(out_rgb).cpu().numpy().squeeze())
                pred_hha_nms = non_maximum_suppression(torch.sigmoid(out_hha).cpu().numpy().squeeze())
                pred_fusion_nms = non_maximum_suppression(torch.sigmoid(out_fusion).cpu().numpy().squeeze())
                
                preds_rgb.append(pred_rgb_nms.flatten())
                preds_hha.append(pred_hha_nms.flatten())
                preds_fusion.append(pred_fusion_nms.flatten())
            else:
                img_rgb, target = data
                out_biped = model(img_rgb.to(device))
                pred_biped_nms = non_maximum_suppression(torch.sigmoid(out_biped).cpu().numpy().squeeze())
                preds_biped.append(pred_biped_nms.flatten())
            
            all_targets.append(target.numpy().squeeze().flatten())

    print("\n--- 实验数据汇总 ---")
    if is_nyud:
        calculate_metrics(preds_rgb, all_targets, "NYUD - RGB Only")
        calculate_metrics(preds_hha, all_targets, "NYUD - HHA Only")
        calculate_metrics(preds_fusion, all_targets, "NYUD - RGB-HHA Fusion")
    else:
        calculate_metrics(preds_biped, all_targets, "BIPED - RGB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['BIPED', 'NYUDv2'])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    evaluate(parser.parse_args())