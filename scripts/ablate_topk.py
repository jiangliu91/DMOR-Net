import argparse
import subprocess
import sys


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    return p.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topks", type=str, default="0,1,2,3,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    for k in topks:
        cmd = [
            sys.executable, "-m", "scripts.train_minimal",
            "--topk", str(k),
            "--seed", str(args.seed),
            "--iters", str(args.iters),
            "--lr", str(args.lr),
            "--batch", str(args.batch),
        ]
        code = run(cmd)
        if code != 0:
            print(f"[ERROR] topk={k} failed with code {code}")
            break

    print("\n✅ ablation done")


if __name__ == "__main__":
    main()
