
# DMOR-EDGE
**Dynamic Modulated Operator Router for Lightweight Edge Detection**

DMOR-EDGE is a lightweight, biologically-inspired edge detection framework that introduces
**dynamic operator routing** into edge detection.  
Instead of relying on fixed parallel pathways, DMOR-EDGE adaptively selects and fuses
multiple lightweight edge-sensitive operators based on image content.

The project is designed for **clean research experiments**, **stable training**, and **reproducible BSDS500 evaluation**.

---

## 🔥 Highlights

- **Dynamic Modulated Operator Router (DMOR)**
  - Operator-space modeling instead of fixed pathways
  - Global + spatial adaptive modulation
  - Top-K sparse routing for competitive selection

- **Lightweight & Efficient**
  - < 1M parameters
  - Depthwise / separable operators
  - No ImageNet pretraining required

- **High-Quality Edge Maps**
  - Clean, thin, and continuous edges
  - Texture suppression without harming recall

- **Full BSDS500 Pipeline**
  - Training → Export → Evaluation
  - CUDA-accelerated ODS / OIS / AP evaluation

---

## 📁 Project Structure

```
DMOR-EDGE/
├─ dataset/
│  └─ BSDS500/
│     ├─ data/
│     │  ├─ images/
│     │  │  ├─ train/
│     │  │  ├─ val/
│     │  │  └─ test/
│     │  └─ groundTruth/
│     │     ├─ train/
│     │     ├─ val/
│     │     └─ test/
│     └─ ucm2/                  # (optional) original BSDS tools
│
├─ models/
│  ├─ dmor.py                   # Dynamic Modulated Operator Router
│  ├─ operators.py              # Edge-sensitive lightweight operators
│  ├─ net.py                    # DMOR-Edge network definition
│  ├─ loss.py                   # Balanced BCE / Dice / Hybrid losses
│  └─ __init__.py
│
├─ scripts/
│  ├─ bsds_train.py             # Training script
│  ├─ bsds_export.py            # Export edge maps (PNG)
│  └─ test_dmor.py              # Quick sanity / debug test
│
├─ pipelines/
│  └─ eval_bsds500.py           # ODS / OIS / AP evaluation (CUDA)
│
├─ outputs/
│  └─ BSDS500/
│     └─ DMOR/
│        ├─ ckpt/               # Trained checkpoints
│        └─ test_png/           # Exported edge maps
│
├─ LICENSE
└─ README.md
```

---

## ⚙️ Environment

- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA ≥ 11.6 (recommended)
- OpenCV, NumPy, SciPy

Install dependencies (example):

```bash
pip install torch torchvision opencv-python numpy scipy
```

---

## 🚀 Training (BSDS500)

Train DMOR-EDGE from scratch:

```bash
python scripts/bsds_train.py \
  --data_root dataset/BSDS500/data \
  --out_dir outputs/BSDS500/DMOR \
  --ckpt_dir outputs/BSDS500/DMOR/ckpt \
  --device cuda \
  --amp
```

- Best checkpoint is saved automatically as:
  ```
  outputs/BSDS500/DMOR/ckpt/dmor_best.pth
  ```

---

## 🖼️ Export Edge Maps

Generate edge prediction PNGs from the trained model:

```bash
python scripts/bsds_export.py \
  --checkpoint outputs/BSDS500/DMOR/ckpt/dmor_best.pth \
  --input_dir dataset/BSDS500/data/images/test \
  --output_dir outputs/BSDS500/DMOR/test_png \
  --device cuda \
  --router_mode dmor \
  --topk 2 \
  --channels 32
```

Output images are normalized, clipped, and suitable for official evaluation.

---

## 📊 Evaluation (ODS / OIS / AP)

Evaluate exported edge maps on BSDS500:

```bash
python pipelines/eval_bsds500.py \
  --pred_dir outputs/BSDS500/DMOR/test_png \
  --gt_dir dataset/BSDS500/data/groundTruth/test \
  --device cuda
```

Example output:

```
ODS: 0.8076 at threshold 0.64
OIS: 0.8157
AP : 0.8507
Precision: 0.7708, Recall: 0.8482
```

---

## 🧠 Method Overview

DMOR-EDGE models **edge detection as operator selection**, not fixed convolution stacks.

- Multiple lightweight operators capture:
  - Difference / contrast
  - Directional structure
  - Multi-scale context
  - Noise suppression

- A **dynamic router** assigns weights:
  - Global (image-level)
  - Spatial (pixel-level)

- **Top-K routing** enforces competition:
  - Sharper edges
  - Less texture noise
  - Better interpretability

---

## 📌 Research Notes

- Designed for **ablation-friendly experiments**
- Easy to extend with new operators
- Routing weights can be visualized for analysis
- Suitable for BSDS500 / BIPED / NYUD

---

## 📜 License

This project is released under the MIT License.

---

## ✨ Acknowledgement

This project is inspired by:
- Biological vision mechanisms (X/Y/W pathways)
- Lightweight edge detection research (PiDiNet, XYW-Net)
- Operator-space modeling and dynamic routing ideas

If you use this code for research, please consider citing appropriately.
