import torch
import argparse
from model import DMOREdge

def export_onnx(args):
    device = torch.device("cpu")
    is_nyud = (args.dataset == 'NYUDv2')
    
    model = DMOREdge(in_channels_rgb=3, in_channels_hha=3 if is_nyud else 0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if is_nyud:
        # NYUDv2 标准输入尺寸
        dummy_rgb = torch.randn(1, 3, 480, 640, device=device)
        dummy_hha = torch.randn(1, 3, 480, 640, device=device)
        inputs = (dummy_rgb, dummy_hha)
        input_names = ["input_rgb", "input_hha"]
        output_names = ["output_rgb", "output_hha", "output_fusion"]
    else:
        # BIPED 高清输入尺寸
        dummy_rgb = torch.randn(1, 3, 720, 1280, device=device)
        inputs = (dummy_rgb, None)
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
    print(f"Successfully exported {args.dataset} model to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="dmor_edge.onnx")
    export_onnx(parser.parse_args())