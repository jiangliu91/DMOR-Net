# DMOR-Edge

**Dynamic Modulated Operator Router for Lightweight Edge Detection**

DMOR-Edge is a lightweight edge-aware feature learning framework that performs **dynamic operator routing** in operator space.  
Instead of fixed multi-branch architectures, DMOR adaptively selects complementary edge-aware operators conditioned on **global image context** and **spatial location**, improving boundary localization while suppressing texture-induced false edges.

---

## Highlights

- **Operator Pool (O1–O5)**: complementary edge-aware operators with different inductive biases  
- **Dynamic Modulated Operator Router (DMOR)**:
  - Global routing (image-level operator preference)
  - Spatial routing (pixel-wise operator selection)
  - **Top-K sparse routing** for efficiency and interpretability
- **Lightweight-ready design**: suitable for compact backbones (<1M parameters target)

---

## Repository Structure

```text
DMOR-Edge/
├─ models/
│  ├─ dmor.py        # DMOR module (routing + Top-K sparse selection)
│  ├─ operators.py   # Operator pool (O1–O5)
│  ├─ net.py         # Minimal end-to-end edge detection network
│  └─ __init__.py
├─ scripts/
│  ├─ test_dmor.py        # DMOR module sanity check
│  └─ train_minimal.py   # Minimal end-to-end training (toy edge task)
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## Environment

Tested on **Windows + Conda**.

### 1) Create conda environment (recommended)

```bash
conda create -n dmore python=3.9 -y
conda activate dmore
```

### 2) Install dependencies

Minimal running requires **PyTorch**.  
Install PyTorch following official instructions for your CUDA / CPU setup.

Verify installation:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Quick Start

### A) DMOR Sanity Check

From the project root:

```bash
python -m scripts.test_dmor
```

Expected output (example):

- Feature shape: `[B, C, H, W]`
- Routing weight shape: `[B, N_ops, H, W]`
- Message: `DMOR training sanity passed`

This verifies:
- Operator pool correctness
- Routing normalization
- Gradient flow through DMOR

---

### B) Minimal End-to-End Training

This script verifies the full training path:

**Backbone → DMOR → Edge Head → Loss → Backpropagation**

```bash
python -m scripts.train_minimal
```

Expected output (example):

```text
iter 010 | loss ...
iter 020 | loss ...
iter 030 | loss ...
...
minimal end-to-end training finished
```

---

## Top-K Sparse Routing Ablation (Proposal-Aligned)

DMOR supports **Top-K sparse routing** via the `topk` parameter.

In `scripts/train_minimal.py`:

```python
model = DMOREdgeNet(channels=32, topk=K).to(device)
```

- `topk = 0` : dense routing (softmax over all operators)  
- `topk = K` : sparse routing (only Top-K operators activated per spatial location)

Recommended ablation settings:

```text
topk ∈ {1, 2, 3, 5}
```

Run:

```bash
python -m scripts.train_minimal
```

This setup is directly aligned with the **Top-K ablation study** described in the proposal.

---

## Notes

- `__pycache__/` directories are ignored via `.gitignore`
- This repository is designed for **research prototyping and ablation studies**

---

## License

MIT
