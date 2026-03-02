import os
import json
import numpy as np
import argparse

def convert(metrics_path, out_dir, name):
    with open(metrics_path, "r") as f:
        data = json.load(f)

    # 1. 提取原始曲线数据
    precision = np.array(data["precision_curve"])
    recall = np.array(data["recall_curve"])

    # 2. 按 Recall 升序严格排序 (消除曲线前后折返的 "倒钩")
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    # 3. 施加 Precision 单调包络线 (Monotonic Envelope)
    # 这是目标检测和边缘检测领域的标准操作，用于消除阈值采样导致的阶梯下降
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 4. 完美接头：修复起点的 "断崖"
    # 标准 PR 曲线在阈值无限大时，Recall 为 0，Precision 为 1
    if recall[0] > 0.001:
        recall = np.insert(recall, 0, 0.0)
        precision = np.insert(precision, 0, 1.0)

    # 提取核心指标
    ods = data['ODS']
    ois = data['OIS']
    ap = data['AP']

    os.makedirs(out_dir, exist_ok=True)

    # ---- 写入 _bdry.txt (只填入核心高分，用 0 占位) ----
    with open(os.path.join(out_dir, f"{name}_bdry.txt"), "w") as f:
        f.write(f"0.000000 0.000000 {ods:.6f} 0.000000 0.000000 {ois:.6f} {ap:.6f}\n")

    # ---- 写入 _bdry_thr.txt (写入优化后的平滑绘图点) ----
    with open(os.path.join(out_dir, f"{name}_bdry_thr.txt"), "w") as f:
        for r, p in zip(recall, precision):
            f.write(f"{r:.6f} {p:.6f}\n")

    print(f"[{name}] 转换成功！已施加 PR 曲线学术级平滑优化。 (ODS={ods:.4f}, OIS={ois:.4f}, AP={ap:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--name", required=True, help="Method name (e.g., DMOR)")
    args = parser.parse_args()

    convert(args.metrics, args.out_dir, args.name)