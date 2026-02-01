# scripts/ablate_topk.py
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    return p.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topks", type=str, default="0,1,2,3,5")
    parser.add_argument("--routers", type=str, default="dmor,uniform")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="runs_minimal")
    args = parser.parse_args()

    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]
    routers = [x.strip() for x in args.routers.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    failures = []
    total = 0
    ok = 0

    for router in routers:
        for seed in seeds:
            for k in topks:
                total += 1
                cmd = [
                    sys.executable, "-m", "scripts.train_minimal",
                    "--router", str(router),
                    "--topk", str(k),
                    "--seed", str(seed),
                    "--iters", str(args.iters),
                    "--lr", str(args.lr),
                    "--batch", str(args.batch),
                    "--save_dir", str(args.save_dir),
                    "--temperature", str(args.temperature),
                ]
                if args.amp:
                    cmd.append("--amp")

                code = run(cmd)
                if code == 0:
                    ok += 1
                else:
                    failures.append((router, k, seed, code))
                    print(f"[ERROR] router={router} topk={k} seed={seed} failed with code {code}")
                    # 不 break：继续跑剩余实验

    print("\n========== ABLATION SUMMARY ==========")
    print(f"Total runs: {total} | Success: {ok} | Failures: {len(failures)}")
    if failures:
        for router, k, seed, code in failures:
            print(f"  - router={router} topk={k} seed={seed} code={code}")
    print("✅ ablation done")


if __name__ == "__main__":
    main()
