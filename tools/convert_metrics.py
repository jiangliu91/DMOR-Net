import os
import json
import numpy as np
import argparse

def convert(metrics_path, out_dir, name):
    with open(metrics_path, "r") as f:
        data = json.load(f)

    precision = np.array(data["precision_curve"])
    recall = np.array(data["recall_curve"])

    # Sort by recall in ascending order to remove back-hooks
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    # Apply monotonic precision envelope (standard in edge detection benchmarks)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Fix cliff at origin: at maximum threshold, recall=0 and precision=1
    if recall[0] > 0.001:
        recall = np.insert(recall, 0, 0.0)
        precision = np.insert(precision, 0, 1.0)

    ods = data['ODS']
    ois = data['OIS']
    ap = data['AP']

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{name}_bdry.txt"), "w") as f:
        f.write(f"0.000000 0.000000 {ods:.6f} 0.000000 0.000000 {ois:.6f} {ap:.6f}\n")

    with open(os.path.join(out_dir, f"{name}_bdry_thr.txt"), "w") as f:
        for r, p in zip(recall, precision):
            f.write(f"{r:.6f} {p:.6f}\n")

    print(f"[{name}] Conversion complete. (ODS={ods:.4f}, OIS={ois:.4f}, AP={ap:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--name", required=True, help="Method name (e.g., DMOR)")
    args = parser.parse_args()

    convert(args.metrics, args.out_dir, args.name)
