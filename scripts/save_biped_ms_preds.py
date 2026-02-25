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
    """多尺度测试 (MS-Test)：SOTA 刷榜的核心秘密"""
    h, w = img.shape[:2]
    all_preds = []
    
    for scale in scales:
        # 正向尺度推断
        img_s = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_s.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        out_s = torch.sigmoid(model(img_t)).cpu().numpy().squeeze()
        out_s = cv2.resize(out_s, (w, h), interpolation=cv2.INTER_LINEAR)
        all_preds.append(out_s)
        
        # 水平翻转推断
        img_flip = img_s[:, ::-1, :]
        img_flip_t = torch.from_numpy(img_flip.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        out_flip = torch.sigmoid(model(img_flip_t)).cpu().numpy().squeeze()
        out_flip = cv2.resize(out_flip[:, ::-1], (w, h), interpolation=cv2.INTER_LINEAR) # 结果翻转回来
        all_preds.append(out_flip)
        
    # 取所有尺度的平均值，极致压制噪点
    final_pred = np.mean(all_preds, axis=0)
    return final_pred

def save_predictions(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在执行多尺度融合推理 (MS-Test) 导出...")
    
    test_img_dir = Path(args.data_root) / 'images' / 'test'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = DMOREdgeNet(channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img_paths = sorted([p for p in test_img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Generating SOTA Edge Maps"):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 使用 MS-Test 获取极致平滑的预测结果
            pred = multi_scale_inference(model, img, device)
            
            # 转换为 uint8 图片直接保存
            pred_uint8 = (pred * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), pred_uint8)
            
    print(f"\n✅ 顶配多尺度预测图片已保存至: {out_dir}")
    print("⚠️ 终极要求：请立刻使用 eval_official_gpu (C++/MATLAB) 对该文件夹进行 SOTA 算分！绝对不要用 Python 算分！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    save_predictions(parser.parse_args())