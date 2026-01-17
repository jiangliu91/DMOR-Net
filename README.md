# DMOR-Edge

Dynamic Modulated Operator Router for Lightweight Edge Detection

## Abstract
We propose DMOR-Edge, a lightweight edge detection framework that performs
dynamic operator rou# DMOR-Edge

Dynamic Modulated Operator Router for Lightweight Edge Detection

---

## Overview
DMOR-Edge is a lightweight edge detection framework that performs **dynamic operator routing
in operator space**. Instead of relying on fixed multi-branch architectures, DMOR-Edge
adaptively selects complementary edge operators conditioned on **global image context**
and **spatial location**, improving boundary localization while effectively suppressing
texture-induced false edges.

---

## Method

### Operator Pool (O1–O5)
We construct a pool of complementary edge-aware operators, each encoding a distinct
inductive bias:

- **O1: Learnable Difference Operator**  
  Enhances local intensity variations and sharp transitions.

- **O2: Center-Difference Convolution (CDC)**  
  Strengthens edge response while maintaining parameter efficiency.

- **O3: Direction-Aware Operator (1×3 + 3×1)**  
  Captures horizontal and vertical edge structures with strong orientation sensitivity.

- **O4: Lightweight Multi-scale Context Operator**  
  Expands receptive fields using dilated convolution for weak and long-range boundaries.

- **O5: Edge-Preserving Smoothing Operator**  
  Suppresses texture noise while preserving structural edges.

---

### Dynamic Modulated Operator Router (DMOR)
Given the operator pool, a **Dynamic Modulated Operator Router (DMOR)** is introduced to
adaptively aggregate operator responses:

- **Global Routing**  
  Learns image-level operator preferences via global context encoding.

- **Spatial Routing**  
  Produces pixel-wise routing weights for location-aware operator selection.

- **Top-K Sparse Routing**  
  Activates only the most relevant operators to improve efficiency and interpretability.

The final output is obtained as a weighted summation of selected operator responses.

---

## Code Structure

DMOR-Edge performs dynamic routing in operator space. Instead of fixed multi-branch
architectures, DMOR-Edge adaptively selects complementary edge operators
according to global image context and spatial location, improving boundary
localization while suppressing texture noise.

architectures, DMOR-Edge adaptively selects complementary edge operators
according to global image context and spatial location, improving boundary
localization while suppressing texture noise.

---

## Quick Start

```python
import torch
from models import DMOR

# Dummy input feature map
x = torch.randn(2, 32, 128, 128)

# Initialize DMOR module
dmor = DMOR(channels=32, topk=2)

# Forward pass
y = dmor(x)
print(y.shape)  # Expected: [2, 32, 128, 128]
```

---

## Method
DMOR-Edge constructs an operator pool consisting of complementary edge-aware
operators, including learnable difference, center-difference convolution,
direction-aware filtering, lightweight multi-scale context, and edge-preserving
smoothing.

A Dynamic Modulated Operator Router (DMOR) is introduced to perform:
- Global routing for image-level operator preference
- Spatial routing for pixel-wise operator selection
- Top-K sparse routing for efficiency and interpretability

## Experiments
Experiments are conducted on BSDS500, BIPED, and NYUD datasets.
Evaluation metrics include ODS, OIS, AP, parameter count, and FLOPs.

## Ablation Studies
We analyze the effect of:
- Global vs spatial routing
- Top-K sparse routing
- Individual operators in the operator pool
- Routing depth and insertion positions

## Status
🚧 Training scripts and pretrained models will be released soon.

## License
MIT
