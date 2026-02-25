# DMOR-Edge

## Dynamic Modulated Operator Router for Lightweight Edge Detection

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)

------------------------------------------------------------------------

# 1. Abstract

DMOR-Edge is a lightweight, operator-space dynamic routing framework for
edge detection. Instead of statically stacking convolutional operators,
DMOR introduces a **Dynamic Modulated Operator Router (DMOR)** that
performs input-adaptive selection and spatially-varying fusion of
complementary lightweight operators.

Unlike conventional lightweight CNNs that rely on fixed receptive
fields, DMOR explicitly models *operator diversity* and performs spatial
routing in operator space.

The framework is designed to:

-   Preserve fine object boundaries
-   Suppress high-frequency texture noise
-   Maintain extremely low parameter count (\<1M)
-   Enable controllable sparsity via Top-K routing
-   Support zero-shot subnetwork pruning from a dense supernet
-   Achieve competitive ODS / OIS / AP with academic-grade evaluation

This repository contains the full implementation, training scripts,
ablation studies, SOTA-aligned evaluation pipelines, and efficiency
profiling tools.

------------------------------------------------------------------------

# 2. Motivation

Traditional lightweight edge detectors rely on:

-   Static convolution stacking
-   Fixed receptive field composition
-   Uniform operator contribution

However, edges in natural images are highly heterogeneous:

-   Structural boundaries require large receptive fields
-   Texture edges require suppression
-   Thin contours require precise gradient modeling
-   Depth edges (NYUDv2) require modality-aware routing

A static operator combination is fundamentally suboptimal.

We instead formulate edge detection as an **operator routing problem**,
where each spatial location dynamically selects the most appropriate
operator subset.

------------------------------------------------------------------------

# 3. Method

## 3.1 Overview

Pipeline:

Input → Lightweight Encoder → DMOR Blocks → Multi-scale Decoder → Edge
Prediction

The key innovation lies in the DMOR block, which performs spatially
varying operator selection and fusion.

------------------------------------------------------------------------

## 3.2 Operator Space Modeling

Given feature tensor:

    F ∈ ℝ^{C×H×W}

We define a diverse set of lightweight operators:

    O = {O1, O2, ..., ON}

Operators include depthwise conv, dilated depthwise conv,
Laplacian-style filters, direction-aware filters, and edge-preserving
smoothing units.

Parallel operator responses:

    Fi = Oi(F)

This creates an operator response space.

------------------------------------------------------------------------

## 3.3 Dynamic Modulated Routing

A lightweight routing network predicts spatial weights:

    W = Softmax( R(F) / T )

Where:

-   R(·) is a compact gating network
-   T is temperature controlling distribution sharpness

Spatially varying weights:

    W ∈ ℝ^{N×H×W}

Final aggregation:

    F_out = Σ_i Wi ⊙ Fi

This enables fine-grained spatial operator modulation.

------------------------------------------------------------------------

## 3.4 Top-K Sparse Routing

To enforce competition and improve efficiency:

For each spatial location:

    Keep only K largest Wi
    Zero out the rest

Benefits:

-   Implicit sparsity
-   Reduced theoretical FLOPs
-   Improved interpretability
-   Zero-shot subnetwork evaluation

This allows evaluating sparse subnetworks directly from a trained dense
supernet checkpoint without retraining.

------------------------------------------------------------------------

## 3.5 Dual-Level Modulation

Global Modulation: - Channel attention across operators

Spatial Modulation: - Pixel-wise routing masks

This dual design balances global structure bias and local adaptivity.

------------------------------------------------------------------------

## 3.6 Decoder & Loss

Lightweight multi-scale decoder:

-   Bilinear upsampling
-   Feature fusion
-   1×1 prediction head
-   Deep supervision (optional)

Hybrid Loss:

    L = λ1 * BCE + λ2 * Dice

For high-resolution datasets (e.g., BIPEDv2), texture-suppression
variants can be enabled.

------------------------------------------------------------------------

# 4. Repository Structure

    DMOR-Edge/
    ├── dataset/
    ├── models/
    ├── scripts/
    ├── pipelines/
    ├── test/
    └── outputs/

Core model files remain immutable. Experimental logic is isolated.

------------------------------------------------------------------------

# 5. Datasets

## 5.1 BSDS500

dataset/BSDS500/data/ ├── images/ └── groundTruth/

## 5.2 BIPEDv2

dataset/BIPEDv2/ ├── imgs/ └── edge_maps/

## 5.3 NYUDv2

dataset/NYUDv2/

Dual-stream RGB-HHA fusion is supported.

------------------------------------------------------------------------

# 6. Training

Example:

    python scripts/bsds_train.py         --data_root dataset/BSDS500/data         --device cuda         --epochs 200         --batch 4         --img_size 512         --router_mode dmor         --topk 2

AMP training is supported.

------------------------------------------------------------------------

# 7. Evaluation (Academic-Grade)

    python pipelines/eval_bsds500.py         --pred_dir outputs/...         --gt_dir dataset/BSDS500/data/groundTruth/test         --ckpt path/to/best.pth

Metrics:

-   ODS
-   OIS
-   AP
-   R50

Supports:

-   Multi-scale testing (MS-Test)
-   Gradient-direction NMS
-   Optimal threshold search
-   Official-style distance tolerance matching

------------------------------------------------------------------------

# 8. Ablation Studies

Located in:

    test/

Includes:

-   Top-K tradeoff
-   Routing strategy comparison
-   Parameter scaling
-   Alpha sensitivity
-   Full operator ablation (B1\~B6)

Overlay injection mechanism ensures core model purity.

------------------------------------------------------------------------

# 9. Efficiency Profiling

    python scripts/test_dmor.py

Outputs:

-   Params (M)
-   FLOPs (G)
-   FPS
-   Theoretical sparse FLOPs (Top-K mode)

------------------------------------------------------------------------

# 10. Reproducibility

-   Fixed seeds
-   Deterministic dataloaders
-   Official evaluation pipeline
-   Structured output tree
-   Fail-fast subprocess experiment runners

------------------------------------------------------------------------

# 11. License

MIT License

------------------------------------------------------------------------

# 12. Citation

@article{dmor_edge_2026, title={Dynamic Modulated Operator Router for
Lightweight Edge Detection}, year={2026} }

------------------------------------------------------------------------

# 13. Updated Repository Notes

-   Master experiment runner
-   Zero-shot Top-K pruning
-   Overlay-based routing experiments
-   Integrated complexity profiling
-   Unified multi-dataset execution

------------------------------------------------------------------------

# 14. Master Runner

    python test/run_everything_dmor_suite.py

Executes:

-   Top-K tradeoff
-   Routing strategy comparison
-   Alpha sensitivity
-   Parameter budget scaling
-   Full ablation suite

------------------------------------------------------------------------

# 15. Overlay Routing Architecture

Located under:

    test/_overlay_dmor/

Design Principles:

-   No modification of models/dmor.py
-   Runtime injection only
-   Clean separation of research variants

------------------------------------------------------------------------

# 16. Output Structure Convention

    outputs/
        ├── ckpt/
        ├── test_png/
        └── eval_official_gpu/

All metrics stored as JSON for reproducibility.

------------------------------------------------------------------------

# 17. Engineering Philosophy

-   Immutable core
-   Overlay-based research isolation
-   Explicit subprocess execution
-   Strict fail-fast behavior
-   Reproducible academic design

------------------------------------------------------------------------
