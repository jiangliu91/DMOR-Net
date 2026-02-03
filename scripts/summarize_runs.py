# scripts/summarize_runs.py
import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List


def flatten(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten lists into compact strings for CSV."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs_minimal", help="directory containing *.json logs")
    ap.add_argument("--out_csv", type=str, default="summary.csv", help="output csv file path")
    args = ap.parse_args()

    pattern = os.path.join(args.runs_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[ERROR] No json logs found under: {pattern}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = os.path.basename(fp)
        rows.append(flatten(data))

    # choose columns (stable + useful)
    preferred = [
        "_file", "router", "topk", "seed",
        "final_loss", "entropy_mean", "confidence_mean", "eff_num_ops_mean",
        "collapse_ratio", "unused_ops",
        "total_params", "dmor_params", "time_sec",
        "winner_ratio_per_op", "topk_membership_ratio_per_op",
        "temperature", "lr", "iters", "batch", "device", "amp",
    ]

    # union all keys
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # keep preferred first, then the rest
    cols = [c for c in preferred if c in all_keys] + sorted([k for k in all_keys if k not in preferred])

    out_path = args.out_csv
    # if out_csv is relative, write under runs_dir for convenience
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.runs_dir, out_path)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"✅ Wrote {len(rows)} rows to: {out_path}")


if __name__ == "__main__":
    main()
