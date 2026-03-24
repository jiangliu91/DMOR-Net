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
    print("=" * 45)
    print("NYUDv2 hardware profiling (480x640)")
    print("=" * 45)

    try:
        from thop import profile
    except ImportError:
        import os
        os.system('pip install thop')
        from thop import profile

    model = DMOREdgeNet(channels=32).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    iters = 50
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
            torch.cuda.synchronize()
    fps = iters / (time.time() - start_time)

    print(f"Params (M): {params / 1e6:.3f}")
    print(f"FLOPs (G):  {flops / 1e9:.2f}")
    print(f"FPS:        {fps:.1f}")
    print("=" * 45)

if __name__ == "__main__":
    measure_nyud_base_metrics()
