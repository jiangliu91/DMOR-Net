
# eval_bsds500.py (with JSON saving)
import os
import cv2
import torch
import numpy as np
import argparse
import json
from glob import glob
from scipy.io import loadmat

def _load_gt_stack(gt_path: str):
    mat = loadmat(gt_path)
    gt = mat["groundTruth"]
    n = gt.shape[1] if gt.shape[0] == 1 else gt.shape[0]
    gts = []
    for i in range(n):
        item = gt[0, i] if gt.shape[0] == 1 else gt[i, 0]
        bnd = item["Boundaries"][0, 0].astype(np.float32)
        gts.append(torch.from_numpy((bnd > 0).astype(np.float32)))
    return torch.stack(gts, dim=0)

def _ap_from_pr(P: torch.Tensor, R: torch.Tensor) -> float:
    order = torch.argsort(R)
    R = R[order]
    P = P[order]
    P_env = P.clone()
    for i in range(P_env.numel() - 2, -1, -1):
        P_env[i] = torch.maximum(P_env[i], P_env[i + 1])
    dR = R[1:] - R[:-1]
    return float(torch.sum(dR * P_env[1:]).item())

def eval_one_image_cuda(pred_png, gt_mat, thresholds, max_dist=0.0075, device="cuda"):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    gts = _load_gt_stack(gt_mat).to(dev)
    pred = cv2.imread(pred_png, 0)
    pred = torch.from_numpy(pred.astype(np.float32) / 255.0).to(dev)

    H, W = pred.shape
    diag = float(np.sqrt(H * H + W * W))
    radius = max_dist * diag
    k = int(2 * radius + 1)
    k = k + 1 if k % 2 == 0 else k
    pad = k // 2

    kernel = torch.ones((1, 1, k, k), device=dev)

    match_gt = torch.nn.functional.conv2d(gts.unsqueeze(1), kernel, padding=pad)
    match_gt = (match_gt > 0).float().squeeze(1)
    match_zone = match_gt.max(dim=0).values

    gt_total = gts.sum(dim=(1, 2)).clamp_min(1.0).sum()

    tp_p = torch.zeros_like(thresholds, device=dev)
    total_p = torch.zeros_like(thresholds, device=dev)
    tp_r = torch.zeros_like(thresholds, device=dev)

    for i, t in enumerate(thresholds):
        p_bin = (pred >= t).float()
        tp_p[i] = (p_bin * match_zone).sum()
        total_p[i] = p_bin.sum().clamp_min(1.0)

        p_d = torch.nn.functional.conv2d(p_bin.unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        p_d = (p_d > 0).float().squeeze()
        tp_r[i] = (gts * p_d).sum(dim=(1, 2)).sum()

    return tp_p, total_p, tp_r, gt_total

def run_eval(pred_dir, gt_dir, device="cuda", threshold_steps=99):
    preds = sorted(glob(os.path.join(pred_dir, "*.png")))
    thresholds = torch.linspace(1.0 / threshold_steps, 1.0 - 1.0 / threshold_steps, threshold_steps)

    total_tp_p = torch.zeros(threshold_steps)
    total_total_p = torch.zeros(threshold_steps)
    total_tp_r = torch.zeros(threshold_steps)
    total_gt_total = torch.tensor(0.0)

    ois_sum, ois_count = 0.0, 0

    for p_path in preds:
        stem = os.path.splitext(os.path.basename(p_path))[0]
        gt_path = os.path.join(gt_dir, stem + ".mat")
        if not os.path.exists(gt_path):
            continue

        tp_p, total_p, tp_r, gt_total = eval_one_image_cuda(
            p_path, gt_path, thresholds, device=device
        )

        total_tp_p += tp_p.cpu()
        total_total_p += total_p.cpu()
        total_tp_r += tp_r.cpu()
        total_gt_total += gt_total.cpu()

        P_i = tp_p / (total_p + 1e-8)
        R_i = tp_r / (gt_total + 1e-8)
        F_i = 2 * P_i * R_i / (P_i + R_i + 1e-8)
        ois_sum += float(torch.max(F_i).item())
        ois_count += 1

    P = total_tp_p / (total_total_p + 1e-8)
    R = total_tp_r / (total_gt_total + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)

    idx = int(torch.argmax(F1).item())
    ods = float(F1[idx].item())
    thr = float((idx + 1) / threshold_steps)
    ois = ois_sum / max(1, ois_count)
    ap = _ap_from_pr(P, R)

    return {
        "ODS": ods,
        "ODS_threshold": thr,
        "OIS": ois,
        "AP": ap,
        "Precision": float(P[idx].item()),
        "Recall": float(R[idx].item()),
        "num_images": ois_count,
        "precision_curve": P.tolist(),
        "recall_curve": R.tolist(),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_dir", default=r"D:\Users\JJzhe\code\github\outputs\BSDS500\DMOR\eval_official_gpu")

    # 鏂板鍙傛暟锛堜笉浼氬奖鍝嶆棫娴佺▼锛?    ap.add_argument("--ckpt", default=None, help="Optional: model checkpoint for complexity profiling")
    ap.add_argument("--ckpt", default=None, help="(optional) ckpt path for Params/FLOPs/FPS")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--topk", type=int, default=2)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        res = run_eval(args.pred_dir, args.gt_dir, device=args.device)
        if args.ckpt:
            try:
                import time
                from thop import profile
                from models.net import DMOREdgeNet

                dev = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
                model = DMOREdgeNet(channels=args.channels, topk=args.topk).to(dev).eval()

                sd = torch.load(args.ckpt, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=False)

                # 1: Params
                params = sum(p.numel() for p in model.parameters())
                res["Params_M"] = params / 1e6

                # 2: FLOPs
                x = torch.randn(1, 3, args.img_size, args.img_size, device=dev)
                flops, _ = profile(model, inputs=(x,), verbose=False)
                res["FLOPs_G"] = flops / 1e9

                # 3: FPS
                warmup, iters = 10, 50
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(x)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(iters):
                        _ = model(x)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.time()
                res["FPS"] = iters / max(1e-9, (t1 - t0))

            except Exception as e:
                print("⚠ Complexity profiling skipped:", e)


    # ----------------------------
    # 鏂板锛歅arams / FLOPs / FPS
    # ----------------------------
    if args.ckpt is not None:
        try:
            from models.net import DMOREdgeNet as EdgeNet
            from thop import profile

            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            model = DMOREdgeNet(channels=args.channels, topk=args.topk)
            sd = torch.load(args.ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
            model.to(device).eval()

            # Params
            total_params = sum(p.numel() for p in model.parameters())
            res["Params_M"] = total_params / 1e6

            # FLOPs
            dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            res["FLOPs_G"] = flops / 1e9

            # FPS
            warmup = 10
            test_iter = 50
            for _ in range(warmup):
                _ = model(dummy)
            torch.cuda.synchronize()
            import time
            start = time.time()
            for _ in range(test_iter):
                _ = model(dummy)
            torch.cuda.synchronize()
            end = time.time()
            fps = test_iter / (end - start)
            res["FPS"] = fps

        except Exception as e:
            print("Complexity profiling skipped:", e)

    # ----------------------------

    print("------------ Evaluation Result ------------")
    for k, v in res.items():
        print(f"{k}: {v}")

    out_path = os.path.join(args.save_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"Saved metrics to: {out_path}")
