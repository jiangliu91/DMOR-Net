# DMOR-Edge

## Dynamic Modulated Operator Router for Lightweight Edge Detection

------------------------------------------------------------------------

# 1. Abstract

DMOR-Edge is a lightweight, operator-space dynamic routing framework for
edge detection. Instead of statically stacking convolutional operators,
DMOR introduces a **Dynamic Modulated Operator Router (DMOR)** that
performs input-adaptive selection and spatially-varying fusion of
complementary lightweight operators.

The framework is designed to:

-   Preserve fine object boundaries
-   Suppress high-frequency texture noise
-   Maintain extremely low parameter count
-   Enable controllable sparsity via Top-K routing
-   Achieve competitive ODS/OIS under \<1M parameters

This repository contains the full implementation, training scripts,
ablation studies, and efficiency profiling tools.

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

A static operator combination is suboptimal.

We instead formulate edge detection as an **operator routing problem**.

------------------------------------------------------------------------

# 3. Method

## 3.1 Overview

Pipeline:

Input → Lightweight Encoder → DMOR Blocks → Multi-scale Decoder → Edge
Prediction

The key innovation lies in the DMOR block.

------------------------------------------------------------------------

## 3.2 Operator Space Modeling

Given feature tensor:

    F ∈ ℝ^{C×H×W}

We define a set of lightweight operators:

    O = {O1, O2, ..., ON}

Each operator is computationally cheap (e.g., DWConv, dilated DWConv,
Laplacian-style filters).

Parallel operator responses:

    Fi = Oi(F)

------------------------------------------------------------------------

## 3.3 Dynamic Routing

A routing network predicts spatial weights:

    W = Softmax( R(F) / T )

Where:

-   R(·) is a lightweight gating network
-   T is temperature parameter

Spatially varying weights:

    W ∈ ℝ^{N×H×W}

Final aggregation:

    F_out = Σ_i Wi ⊙ Fi

------------------------------------------------------------------------

## 3.4 Top-K Sparse Routing

To encourage competition and reduce redundancy:

For each spatial location:

    Keep only K largest Wi
    Set others to zero

This introduces:

-   Implicit sparsity
-   Improved interpretability
-   Reduced effective computation

------------------------------------------------------------------------

## 3.5 Dual-Level Modulation

Global Modulation: - Channel attention over operators

Spatial Modulation: - Pixel-wise routing mask

This dual modulation enables both:

-   Global structural bias
-   Local adaptive selection

------------------------------------------------------------------------

## 3.6 Decoder

Lightweight multi-scale decoder:

-   Bilinear upsampling
-   Feature fusion
-   1×1 prediction head

Loss:

    L = λ1 * BCE + λ2 * Dice

------------------------------------------------------------------------

# 4. Repository Structure

    DMOR-Edge/
    ├── dataset/
    │   ├── BSDS500/
    │   ├── BIPEDv2/
    │   └── NYUDv2/
    │
    ├── models/
    │   ├── dmor.py
    │   ├── backbone.py
    │   └── ...
    │
    ├── scripts/
    │   ├── bsds_train.py
    │   ├── bsds_export.py
    │   ├── biped_train.py
    │   ├── biped_export.py
    │   ├── nyudv2_train.py
    │   ├── nyudv2_export.py
    │   ├── test_dmor.py
    │   └── _dataset_common.py
    │
    ├── pipelines/
    │   ├── eval_bsds500.py
    │   ├── eval_biped.py
    │   ├── eval_nyudv2.py
    │   └── eval_generic_png.py
    │
    ├── test/
    │   ├── run_ablation_bsds500_suite.py
    │   ├── run_alpha_sensitivity_bsds500.py
    │   ├── run_param_budget_scaling_bsds500.py
    │   ├── run_routing_strategy_bsds500.py
    │   ├── run_topk_tradeoff_bsds500.py
    │   └── _overlay_dmor*

------------------------------------------------------------------------

# 5. Datasets

## 5.1 BSDS500

Download:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

Structure:

dataset/BSDS500/data/ ├── images/ └── groundTruth/

------------------------------------------------------------------------

## 5.2 BIPEDv2

Download: https://github.com/xavysp/BIPED

Expected:

dataset/BIPEDv2/ ├── imgs/ └── edge_maps/

------------------------------------------------------------------------

## 5.3 NYUDv2

Download: https://cs.nyu.edu/\~silberman/datasets/nyu_depth_v2.html

Place .mat in:

dataset/NYUDv2/

Preprocess:

    python scripts/nyudv2_export.py

------------------------------------------------------------------------

# 6. Training

Example (BSDS500):

    python scripts/bsds_train.py \
        --data_root dataset/BSDS500/data \
        --device cuda \
        --epochs 200 \
        --batch 4 \
        --img_size 512 \
        --router_mode dmor \
        --topk 2

------------------------------------------------------------------------

# 7. Evaluation

    python pipelines/eval_bsds500.py \
        --pred_dir outputs/... \
        --gt_dir dataset/BSDS500/data/groundTruth/test

Metrics:

-   ODS
-   OIS
-   AP
-   R50

------------------------------------------------------------------------

# 8. Ablation Studies

Located in:

    test/

Includes:

-   Top-K tradeoff
-   Routing strategy
-   Parameter scaling
-   Alpha sensitivity

------------------------------------------------------------------------

# 9. Efficiency Profiling

    python scripts/test_dmor.py

Outputs:

-   Params (M)
-   FLOPs (G)
-   FPS

------------------------------------------------------------------------

# 10. Reproducibility

-   Fixed random seeds
-   Official evaluation scripts
-   Structured output directory
-   Fully self-contained training pipelines

------------------------------------------------------------------------

# 11. License

MIT License

------------------------------------------------------------------------

# 12. Citation

Dynamic Modulated Operator Router for Lightweight Edge Detection

------------------------------------------------------------------------

# 13. Updated Repository Notes (New Additions)

The current repository additionally supports:

-   Master experiment runner (multi-dataset automation)
-   Alpha overlay routing experiments
-   Automatic checkpoint resolution
-   Integrated complexity profiling during evaluation
-   Unified BSDS experiment suite execution

------------------------------------------------------------------------

# 14. Master Runner (Full Automation)

You can run all BSDS experiment suites and multi-dataset pipelines
using:

    python test/run_everything_dmor_suite.py

Optional arguments:

    --alphas 0.0,0.5,1.0
    --variants tiny,small
    --channels_map tiny:16,small:32
    --device cuda
    --epochs 200
    --batch 4

This script sequentially executes:

-   Top-K tradeoff
-   Routing strategy comparison
-   Alpha sensitivity
-   Parameter budget scaling
-   Full ablation suite

------------------------------------------------------------------------

# 15. Alpha Sensitivity (Overlay Routing)

Alpha-based modulation experiments are isolated under:

    test/_overlay_dmor/

The overlay mechanism ensures:

-   Core DMOR logic in models/dmor.py remains untouched
-   Alpha experiments do not modify the base architecture
-   Clean separation between research variants and main model

------------------------------------------------------------------------

# 16. Integrated Complexity Profiling

Efficiency metrics (Params / FLOPs / FPS) can be computed during
evaluation by passing:

    --ckpt path/to/best.pth
    --img_size 512
    --channels 32
    --topk 2

Example:

    python -m pipelines.eval_bsds500         --pred_dir outputs/...         --gt_dir dataset/BSDS500/data/groundTruth/test         --ckpt outputs/.../best.pth         --img_size 512         --channels 32         --topk 2

Additional metrics automatically added:

-   Params_M
-   FLOPs_G
-   FPS

------------------------------------------------------------------------

# 17. Output Structure Convention

All experiment outputs follow:

    outputs/
        ├── BSDS500/
        ├── BIPEDv2/
        └── NYUDv2/

Each dataset contains:

    ckpt/
    test_png/
    eval_official_gpu/

Experiment summaries are saved as JSON for reproducibility.

------------------------------------------------------------------------

# 18. Engineering Principles (Extended)

-   Core model files are never deleted --- only extended
-   Experiment logic isolated in test/
-   Overlay modules do not pollute base model
-   All experiment runners use subprocess with explicit cwd
-   Strict fail-fast behavior for reproducibility
