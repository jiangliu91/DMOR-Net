# DMOR-Edge

Dynamic Modulated Operator Router for Lightweight Edge Detection

## Abstract
We propose DMOR-Edge, a lightweight edge detection framework that performs
dynamic operator routing in operator space. Instead of fixed multi-branch
architectures, DMOR-Edge adaptively selects complementary edge operators
according to global image context and spatial location, improving boundary
localization while suppressing texture noise.

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
