# run_bsds500_eval_conda.py
"""
One-command BSDS500 final step from your CONDA env:
1) Export predictions to PNG via scripts/bsds_export.py
2) Run MATLAB official BSDS evaluation (edgesEvalDir) in batch mode

Run (from repo root):
  python run_bsds500_eval_conda.py

Optional args:
  --ckpt <path_to_dmor_best.pth>
  --matlab_exe "<full_path_to_matlab.exe>"
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def find_matlab_exe(user_given: str | None) -> str:
    if user_given:
        p = Path(user_given)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"--matlab_exe not found: {user_given}")

    env = os.environ.get("MATLAB_EXE", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"Env MATLAB_EXE points to missing file: {env}")

    # Try common Windows locations and pick newest
    candidates = []
    for base in [r"C:\Program Files\MATLAB", r"C:\Program Files (x86)\MATLAB"]:
        b = Path(base)
        if b.is_dir():
            candidates += [str(p) for p in b.glob(r"R*\bin\matlab.exe")]

    if not candidates:
        raise FileNotFoundError(
            "Cannot find matlab.exe automatically.\n"
            "Fix: pass --matlab_exe \"C:\\Program Files\\MATLAB\\R2023b\\bin\\matlab.exe\" "
            "or set env MATLAB_EXE."
        )

    def ver_key(p: str):
        # ...\MATLAB\R2023b\bin\matlab.exe
        tag = Path(p).parts[-3]  # R2023b
        import re
        m = re.match(r"R(\d{4})([ab])", tag)
        return (int(m.group(1)), m.group(2)) if m else (0, "a")

    candidates.sort(key=ver_key, reverse=True)
    return candidates[0]


def write_eval_m(m_file: Path, bsds_root: str, pred_dir: str, out_dir: str):
    # Try to locate bench code under dataset
    bench1 = Path(bsds_root) / "BSR" / "bench"
    bench2 = Path(bsds_root) / "bench"

    if bench1.is_dir():
        bench_path = str(bench1)
    elif bench2.is_dir():
        bench_path = str(bench2)
    else:
        bench_path = ""  # user must add manually

    m_text = f"""% Auto-generated: BSDS500 evaluation (edgesEvalDir)
bsdsRoot = '{bsds_root}';
predDir  = '{pred_dir}';
outDir   = '{out_dir}';

gt1 = fullfile(bsdsRoot, 'groundTruth', 'test');
gt2 = fullfile(bsdsRoot, 'ground_truth', 'test');
if exist(gt1, 'dir'); gtDir = gt1;
elseif exist(gt2, 'dir'); gtDir = gt2;
else
    error('Cannot find groundTruth/ground_truth under BSDS root');
end

mkdir(outDir);

% Add BSDS benchmark code (edgesEvalDir)
"""
    if bench_path:
        m_text += f"addpath(genpath('{bench_path}'));\n"
    else:
        m_text += "% TODO: addpath(genpath('D:\\path\\to\\BSR\\bench'));\n"
    m_text += """
edgesEvalDir(predDir, gtDir, 'outDir', outDir);

disp('===== eval_bdry.txt =====');
type(fullfile(outDir, 'eval_bdry.txt'));
"""
    m_file.write_text(m_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=r"D:\Users\JJzhe\code\github\DMOR-Edge")
    parser.add_argument("--bsds_root", type=str, default=r"D:\Users\JJzhe\code\github\dataset\BSDS500")
    parser.add_argument("--out_root",  type=str, default=r"D:\Users\JJzhe\code\github\outputs\BSDS500\DMOR")
    parser.add_argument("--ckpt", type=str, default=r"D:\Users\JJzhe\code\github\outputs\BSDS500\DMOR\ckpt\dmor_best.pth")
    parser.add_argument("--matlab_exe", type=str, default="")
    parser.add_argument("--backbone", type=str, default="lite", choices=["tiny", "lite"])
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--router", type=str, default="dmor", choices=["dmor", "uniform"])
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")

    # 1) export png
    pred_dir = Path(args.out_root) / "test_png"
    pred_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BSDS_ROOT"] = args.bsds_root
    env["OUT_DIR"] = str(pred_dir)
    env["CKPT_PATH"] = args.ckpt if Path(args.ckpt).is_file() else ""  # if missing, still run pipeline
    env["BACKBONE"] = args.backbone
    env["TOPK"] = str(args.topk)
    env["ROUTER_MODE"] = args.router
    env["TEMPERATURE"] = str(args.temperature)

    print("[1/2] Exporting test_png ...")
    subprocess.run([sys.executable, "-m", "scripts.bsds_export"], cwd=str(repo_root), env=env, check=True)

    # 2) matlab eval
    matlab_exe = find_matlab_exe(args.matlab_exe if args.matlab_exe else None)

    eval_dir = Path(args.out_root) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    m_file = repo_root / "scripts" / "eval_bsds.m"
    write_eval_m(m_file, args.bsds_root, str(pred_dir), str(eval_dir))

    print("[2/2] Running MATLAB edgesEvalDir ...")
    subprocess.run([matlab_exe, "-batch", f"run('{str(m_file)}')"], check=True)

    res = eval_dir / "eval_bdry.txt"
    print("\n✅ Finished. Check:")
    print(f"  {res}")


if __name__ == "__main__":
    main()
