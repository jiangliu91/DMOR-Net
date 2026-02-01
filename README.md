# DMOR-Edge
Dynamic Modulated Operator Router for Lightweight Edge Detection

## Overview
DMOR-Edge introduces a **Dynamic Modulated Operator Router (DMOR)** for lightweight edge detection.
Instead of relying on a single fixed operator, DMOR adaptively combines a small pool of complementary
lightweight operators using **learned, spatially-varying routing weights**.

This repository is intentionally designed for **clean ablation, interpretability, and reproducibility**,
rather than leaderboard-oriented optimization.

Key characteristics:
- Dynamic operator routing with global + spatial modulation
- Optional **Top-K sparse routing** for controlled sparsity
- A strict **No-Router (uniform) baseline** for fair comparison
- Lightweight design (<1M parameters)
- Simple CLI-based experimental workflow

---

## Method: Dynamic Modulated Operator Router (DMOR)

Given an input feature map, DMOR performs the following steps:

1. Apply a pool of predefined lightweight operators
2. Generate routing logits via global and spatial routers
3. Normalize routing weights using softmax
4. Optionally enforce **Top-K sparsity** on routing weights
5. Aggregate operator outputs by weighted summation

### No-Router (Uniform) Baseline
To isolate the contribution of routing, DMOR-Edge provides a **uniform baseline** in which:
- No routing network is used
- No Top-K masking is applied
- All operators are equally weighted (1 / N)

This baseline serves as a principled reference for ablation studies and reviewer-facing analysis.

---

## Repository Structure

```text
DMOR-Edge/
├── models/
│   ├── __init__.py
│   ├── operators.py      # Lightweight operator pool
│   ├── dmor.py           # DMOR routing module (dmor / uniform)
│   └── net.py            # Tiny backbone + DMOR + edge head
├── scripts/
│   ├── test_dmor.py      # Training sanity check
│   ├── train_minimal.py  # Minimal end-to-end training & routing analysis
│   └── ablate_topk.py    # Top-K / router / seed ablation sweep
├── README.md
└── LICENSE
```

---

## Installation

This project depends only on PyTorch and standard Python packages.

```bash
conda create -n dmore python=3.9 -y
conda activate dmore
pip install torch torchvision
```

---

## Quick Start

Run a basic sanity check to verify forward/backward correctness:

```bash
python scripts/test_dmor.py
```

Expected output:
```
✅ DMOR training sanity passed
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
  --seed 0
```

### 2. No-Router (Uniform) Baseline
```bash
python -m scripts.train_minimal \
  --router uniform \
  --topk 0 \
  --iters 5000 \
  --lr 1e-3 \
  --batch 4 \
  --seed 0
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
- `1, 2, 3, 5` : sparse Top-K routing

---

## Logged Metrics

Each training run records the following statistics:
- Final training loss
- Routing entropy
- Routing confidence (mean max probability)
- Effective number of active operators
- Operator selection distribution (Top-1 and Top-K)
- Parameter counts
- Runtime

All metrics are saved in both `.txt` and `.json` formats and are suitable for direct inclusion
in ablation tables and figures.

---

## Notes

- Training scripts use a toy edge dataset for sanity and controlled ablation analysis.
- The framework is designed to be easily extended to real-world edge detection datasets
  (e.g., BSDS500, BIPED, NYUD).

---

## License

This project is released under the MIT License.
