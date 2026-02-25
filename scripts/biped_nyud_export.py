import os
import sys
import torch
import argparse
from pathlib import Path

# 获取项目根目录 (DMOR-Edge) 并加入系统路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 修正导入路径：从 models 导入网络，从 scripts 导入 Wrapper
from models.net import DMOREdgeNet
from scripts.biped_nyud_train import DMORFusionWrapper

def export_onnx(args):
    device = torch.device("cpu")
    is_nyud = (args.dataset == 'NYUDv2')
    
    if is_nyud:
        model = DMORFusionWrapper(channels=args.channels).to(device)
    else:
        model = DMOREdgeNet(channels=args.channels).to(device)
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if is_nyud:
        # NYUDv2 standard inference size
        dummy_rgb = torch.randn(1, 3, 480, 640, device=device)
        dummy_hha = torch.randn(1, 3, 480, 640, device=device)
        inputs = (dummy_rgb, dummy_hha)
        input_names = ["input_rgb", "input_hha"]
        output_names = ["output_rgb", "output_hha", "output_fusion"]
    else:
        # BIPED high-resolution inference size
        dummy_rgb = torch.randn(1, 3, 720, 1280, device=device)
        inputs = (dummy_rgb,)
        input_names = ["input_rgb"]
        output_names = ["output_edge"]

    torch.onnx.export(
        model,
        inputs,
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch_size", 2: "height", 3: "width"} for name in input_names + output_names}
    )
    print(f"Successfully exported {args.dataset} mathematical graph model to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['BIPED', 'NYUDv2'])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="dmor_edge.onnx")
    parser.add_argument("--channels", type=int, default=32)
    export_onnx(parser.parse_args())