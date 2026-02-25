import torch
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.net import DMOREdgeNet

def measure_nyud_base_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*45)
    print("开始测算 NYUDv2 基础网络硬件指标 (分辨率 480x640)")
    print("="*45)

    try:
        from thop import profile
    except ImportError:
        import os
        os.system('pip install thop')
        from thop import profile

    # 只初始化单路网络，对齐 RGB/HHA 的真实参数
    model = DMOREdgeNet(channels=32).to(device)
    model.eval()

    # NYUDv2 标准输入尺寸
    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    # 1. 测算 Params 和 FLOPs
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)

    # 2. 测算 FPS
    for _ in range(10): # GPU 预热
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    start_time = time.time()
    iters = 50
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
            torch.cuda.synchronize()
    fps = iters / (time.time() - start_time)

    # 打印最终结果，直接填入表格第一行 (RGB)
    print(f"Params (M): {params / 1e6:.3f}")
    print(f"FLOPs (G):  {flops / 1e9:.2f}")
    print(f"FPS:        {fps:.1f}")
    print("="*45)

if __name__ == "__main__":
    measure_nyud_base_metrics()