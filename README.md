# DMOR-Edge
**Dynamic Modulated Operator Router for Lightweight Edge Detection**

DMOR-Edge introduces a **Dynamic Modulated Operator Router (DMOR)** for lightweight edge detection.
The core idea is to dynamically route and combine multiple lightweight operators using
content-adaptive, spatially-varying weights, achieving strong performance under a strict
parameter budget.

This repository is designed with **research reproducibility and interpretability** in mind,
and provides a **fully automated pipeline** for training, inference, and **official BSDS500
evaluation**.

---

## ✨ Key Features

- **Dynamic Modulated Operator Router (DMOR)**
  - Spatially-varying, content-adaptive operator routing
  - Top-K sparse routing support
  - Explicit No-Router (uniform) baseline for fair comparison

- **Lightweight Design**
  - Parameter budget < 1M
  - Operator pool built from efficient local operators

- **Official BSDS500 Evaluation**
  - MATLAB **BSDSBench** (`edgesEvalDir`, `edgesEvalImg`)
  - Metrics: **ODS / OIS / AP**
  - Parallel evaluation via `fevalDistr (parfor)`
  - Fully automated from Python (no manual MATLAB interaction)

---

## 📁 Repository Structure

```text
DMOR-Edge
├── dataset/
│   └── BSDS500/
│       ├── data/                 # Official BSDS500 data (images, groundTruth)
│       └── README.md
├── models/
│   ├── dmor.py                   # Dynamic Modulated Operator Router
│   ├── operators.py              # Lightweight operator pool
│   └── net.py                    # Backbone + DMOR + edge head
├── pipelines/
│   └── run_bsds500_pipeline.py   # One-shot BSDS500 evaluation pipeline
├── scripts/
│   ├── bsds_train.py             # Training script
│   ├── bsds_export.py            # Export predictions to PNG
│   ├── ablate_topk.py            # Top-K routing ablation
│   └── summarize_runs.py         # Result summarization
├── tools/
│   ├── BSDSbench/                # Official BSDS500 MATLAB benchmark
│   ├── pdollar_toolbox/          # Piotr Dollar MATLAB toolbox
│   └── eval_bsds500_official.m   # Auto-generated MATLAB eval script
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Environment

- Python ≥ 3.8
- PyTorch ≥ 1.10
- MATLAB **R2020+**
  - **Image Processing Toolbox** (required for `bwmorph`)
  - Parallel Computing Toolbox (optional, for `parfor`)

---

### 2. BSDS500 Dataset

Place the BSDS500 dataset in the following structure:

```text
dataset/BSDS500/data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── groundTruth/
    ├── train/
    ├── val/
    └── test/
```

---

### 3. Training (optional)

```bash
python scripts/bsds_train.py
```

---

### 4. Export Predictions

Export edge maps as PNG files for official evaluation:

```bash
python scripts/bsds_export.py
```

This will generate a directory such as:

```text
outputs/BSDS500/DMOR/test_png/
```

---

## 📊 Official BSDS500 Evaluation

This project uses the **official BSDS500 MATLAB benchmark (BSDSBench)**.

### Evaluation Protocol

- Evaluator: `edgesEvalDir.m`, `edgesEvalImg.m`
- Metrics: **ODS / OIS / AP**
- Default settings:
  - `thin = 1`
  - `maxDist = 0.0075`
  - `thrs = 99`
- Parallel execution via `fevalDistr (parfor)`

### One-shot Evaluation (Recommended)

```bash
python pipelines/run_bsds500_pipeline.py   --repo_root <path-to-DMOR-Edge>   --bsds_root <path-to-BSDS500/data>   --matlab "<path-to-matlab.exe>"   --pred_dir <prediction_png_dir>   --out_dir <output_dir>   --eval_type parfor
```

> **Note**
> - MATLAB **Image Processing Toolbox** is required.
> - Use `--eval_type local` if Parallel Computing Toolbox is unavailable.

Evaluation results will be saved to:

```text
outputs/BSDS500/DMOR/eval_official/
```

---

## 🧪 Ablation & Analysis

- **Top-K Routing Ablation**
  ```bash
  python scripts/ablate_topk.py
  ```

- **Result Summary**
  ```bash
  python scripts/summarize_runs.py
  ```

---

## 📝 Development Log

- **2026-02**
  - Finalized BSDS500 official evaluation pipeline
  - Verified MATLAB BSDSBench + `fevalDistr` integration
  - Fixed path isolation and toolbox dependency issues

---

## 📦 Datasets and Third-Party Components (Attribution)

This repository contains **non-original** components for research reproducibility:

### BSDS500 Dataset
- Name: Berkeley Segmentation Dataset and Benchmark (BSDS500)
- Purpose: Edge detection evaluation (ODS / OIS / AP)
- Source: Berkeley Vision Group (official BSDS resources)

### MATLAB BSDSBench (Official Evaluation Code)
- Purpose: Official BSDS500 evaluation (`edgesEvalDir`, `edgesEvalImg`, etc.)
- Source: Provided by the BSDS authors / Berkeley Vision Group

### Piotr Dollár MATLAB Toolbox
- Purpose: Utility toolbox required by BSDSBench (`fevalDistr`, etc.)
- Author: Piotr Dollár

All datasets and third-party code remain the property of their respective owners.
They are included here for **research and reproducibility**. If any licensing terms
require additional attribution or redistribution restrictions, please refer to the
original sources and licenses shipped with those materials.

## 📄 License

This project is released under the **MIT License**.
