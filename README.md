# DMOR-Edge
Dynamic Modulated Operator Router for Lightweight Edge Detection

## Overview
DMOR-Edge introduces a **Dynamic Modulated Operator Router (DMOR)** that adaptively combines multiple lightweight operators for edge detection.  
The router dynamically assigns spatially-varying weights to operators, enabling content-adaptive computation with minimal parameter overhead.

This repository focuses on **clean ablation design and reproducibility**, including:
- Top-K sparse routing ablations
- A **No-Router (uniform) baseline** for fair comparison
- Lightweight parameter budget (<1M parameters)
- Simple CLI-based experimental workflow

---

## Method: DMOR
Given an input feature map, DMOR:
1. Applies a pool of predefined lightweight operators
2. Uses global + spatial routers to generate routing logits
3. Normalizes routing weights via softmax
4. Optionally enforces **Top-K sparsity**
5. Aggregates operator outputs using weighted summation

### No-Router Baseline
To isolate the contribution of routing, we also provide a **uniform (no-router) baseline**, where all operators are equally weighted:
- No routing network
- No Top-K masking
- Equal weights: 1 / N

This baseline is essential for principled ablation and reviewer-facing analysis.

---

## Repository Structure
```
DMOR-Edge/
├── models/
│   ├── dmor.py          # DMOR block (routing / uniform baseline)
│   ├── net.py           # Edge detection network
│   └── operators.py    # Lightweight operator pool
├── scripts/
│   ├── train_minimal.py # Minimal training & ablation script
│   ├── ablate_topk.py   # Top-K routing ablation
│   └── test_dmor.py     # Sanity check
├── README.md
└── LICENSE
```

---

## Installation
This code depends only on PyTorch and standard Python packages.

```bash
conda create -n dmore python=3.9
conda activate dmore
pip install torch torchvision
```

---

## Quick Start
Run a basic sanity check:
```bash
python scripts/test_dmor.py
```

---

## Training & Ablations

### 1. DMOR with Dense Routing
```bash
python -m scripts.train_minimal \
  --router dmor \
  --topk 0 \
  --iters 5000 \
  --lr 1e-3 \
  --batch 4 \
  --seed 0 \
  --device cpu \
  --save_dir runs/router_ablate/dmor_dense
```

### 2. No-Router (Uniform) Baseline
```bash
python -m scripts.train_minimal \
  --router uniform \
  --topk 0 \
  --iters 5000 \
  --lr 1e-3 \
  --batch 4 \
  --seed 0 \
  --device cpu \
  --save_dir runs/router_ablate/uniform
```

### 3. Top-K Sparse Routing
```bash
python -m scripts.train_minimal \
  --router dmor \
  --topk 2 \
  --iters 5000 \
  --lr 1e-3 \
  --batch 4
```

Supported `--topk` values:
- `0` : dense routing
- `1,2,3,5` : sparse Top-K routing

---

## Logged Metrics
Each run records:
- Final training loss
- Routing entropy
- Operator selection distribution
- Parameter counts
- Runtime

These statistics are designed for **direct inclusion in ablation tables**.

---

## Key Observations
- Learned routing consistently reduces routing entropy compared to the uniform baseline
- Top-K routing enforces sparsity while preserving performance
- The uniform baseline reaches theoretical maximum entropy (log N), validating experimental correctness

---

## Notes
- Training scripts use a toy edge dataset for sanity and ablation analysis
- The framework is designed to be easily extended to real-world datasets (e.g., BSDS, NYUD)

---

## License
This project is released under the MIT License.
