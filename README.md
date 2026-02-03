# DMOR-Edge
Dynamic Modulated Operator Router for Lightweight Edge Detection

## Overview
DMOR-Edge introduces a **Dynamic Modulated Operator Router (DMOR)** that adaptively combines multiple
lightweight operators for edge detection.
The router assigns **spatially-varying, content-adaptive weights** to operators, enabling flexible
feature modulation with minimal parameter overhead.

This repository focuses on **clean ablation design, interpretability, and reproducibility**, including:
- Top-K sparse routing with proper renormalization
- A **No-Router (uniform) baseline** for fair comparison
- Lightweight parameter budget (<1M parameters)
- Explicit routing diagnostics and sanity checks
- Simple, script-based experimental workflow

---

## Method: DMOR
Given an input feature map, DMOR performs the following steps:

1. **Operator Pool Evaluation**  
   A fixed pool of lightweight, resolution-preserving operators is applied in parallel.

2. **Routing Logit Generation**  
   - A **global router** produces image-level operator priors  
   - A **spatial router** produces pixel-wise routing logits

3. **Weight Normalization**  
   Routing logits are combined, temperature-scaled, and normalized via softmax.

4. **Optional Top-K Sparsification**  
   - Per-pixel Top-K masking is applied  
   - Remaining weights are **renormalized** to ensure numerical correctness

5. **Weighted Fusion**  
   Operator outputs are aggregated using a memory-efficient weighted summation
   (without stacking large intermediate tensors).

### No-Router Baseline
To isolate the contribution of routing, a **uniform (no-router) baseline** is provided:
- No routing network
- No Top-K masking
- Equal weights (1 / N) for all operators

This baseline is critical for principled ablation and reviewer-facing comparisons.

---

## Repository Structure
```
DMOR-Edge/
├── models/
│   ├── __init__.py        # Public API and versioning
│   ├── dmor.py            # DMOR routing & fusion block
│   ├── net.py             # Minimal end-to-end edge network
│   └── operators.py      # Lightweight operator pool
├── scripts/
│   ├── train_minimal.py  # Minimal training & routing diagnostics
│   ├── ablate_topk.py    # Systematic Top-K ablation runner
│   ├── test_dmor.py      # Training-level sanity checks (incl. Top-K)
│   └── summarize_runs.py # JSON → CSV ablation summarization
├── README.md
└── LICENSE
```

---

## Installation
This code depends only on PyTorch and standard Python packages.

```bash
conda create -n dmore python=3.9
conda activate dmore
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## Quick Start
Run a training-level sanity check (forward, backward, Top-K correctness):

```bash
python scripts/test_dmor.py
```

---

## Training & Ablations

### 1. DMOR with Dense Routing
```bash
python -m scripts.train_minimal   --router dmor   --topk 0   --iters 5000   --lr 1e-3   --batch 4   --seed 0   --device gpu   --save_dir runs/router_ablate/dmor_dense
```

### 2. No-Router (Uniform) Baseline
```bash
python -m scripts.train_minimal   --router uniform   --topk 0   --iters 5000   --lr 1e-3   --batch 4   --seed 0   --device gpu   --save_dir runs/router_ablate/uniform
```

### 3. Top-K Sparse Routing
```bash
python -m scripts.train_minimal   --router dmor   --topk 2   --iters 5000   --lr 1e-3   --batch 4   --device gpu
```

Supported `--topk` values:
- `0` : dense routing
- `1, 2, 3, 5` : sparse Top-K routing

---

## Logged Metrics
Each run records detailed routing diagnostics:
- Final training loss
- Routing entropy
- Confidence (maximum routing probability)
- Effective number of operators (`exp(entropy)`)
- Top-1 operator winner ratio
- Top-K membership ratio per operator
- Collapse ratio and unused-operator count
- Parameter counts and runtime

These statistics are designed for **direct inclusion in ablation tables and plots**.

---

## Automatic Ablation Summarization
After running multiple experiments, all results can be aggregated automatically:

```bash
python -m scripts.summarize_runs --runs_dir runs_minimal --out_csv summary.csv
```

This generates a single CSV file suitable for spreadsheet analysis or LaTeX tables.

---

## Notes
- Default experiments use a toy edge dataset to isolate routing behavior
- The framework is designed to be easily extended to real datasets
  (e.g., BSDS500, NYUD-v2)
- A stronger lightweight backbone is available for real-data experiments

---

## License
This project is released under the MIT License.
