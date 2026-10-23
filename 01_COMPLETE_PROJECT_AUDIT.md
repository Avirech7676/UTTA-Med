# 01. COMPLETE PROJECT AUDIT & REPOSITORY MANIFEST
**UTTA-Med Project 2 — Comprehensive Research Development & Verification Audit**  
**Date:** October 2026 | **Working Branch:** `research-project-2`  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Auditor:** Senior Deep Learning Researcher, Reproducibility Auditor & Computational Pathology Specialist

---

## 1. EXECUTIVE SUMMARY & WORKING BRANCH INITIALIZATION

In compliance with Phase 1 instructions:
1. **Working Branch:** Created and checked out active research branch:
   ```bash
   git checkout -b research-project-2
   ```
2. **Preservation of Research Artifacts:** All previously verified source checkpoints, feature centroids, prediction arrays, and evaluation manifests have been inventoried and preserved.
3. **Synthetic Experiment Isolation:** The previously identified synthetic experiments have been verified as isolated within `archive/synthetic_v2_unverified/` with explicit warning notices (`ARCHIVE_NOTICE.md`). They cannot enter the production execution or evaluation pipeline.
4. **Repository Manifest:** A comprehensive inventory of all 874 key repository files, covering architecture, source code, checkpoints, configurations, logs, and results, has been constructed with cryptographic SHA-256 checksums and functional role annotations.

---

## 2. REPOSITORY ARCHITECTURE & SUBSYSTEM AUDIT

### 2.1 Subsystem Overview

```
UTTA_MED_V1_FINAL/
├── src/                          # Core production libraries
│   ├── data/                     # Dataset loaders (Parquet, WILDS, MIDOG)
│   ├── models/                   # Neural network architectures (ResNet-18 with Head Dropout)
│   └── v2/                       # UTTA-Med 2.0 & Bio-SafeAdapt core modules
│       ├── gating/               # Adaptive uncertainty & multi-signal gating engines
│       ├── safeadapt/            # Bio-SafeAdapt controllers, monitors & elastic anchoring
│       └── utils/                # Coordinate transforms, metric logging, checkpoint IO
├── scripts/                      # Autonomous execution, evaluation & validation pipelines
│   ├── kaggle/                   # Automated Kaggle GPU orchestration & worker scripts
│   └── v2/                       # Calibration and multi-stage evaluation runners
├── checkpoints/                  # Cryptographically verified PyTorch model weights & centroids
├── configs/                      # YAML configuration files for classification & detection
├── data_manifests/               # Canonical dataset manifests (Camelyon17 & MIDOG)
├── results/                      # Empirical prediction outputs, bootstrap arrays & metrics
├── figures/                      # Publication-quality vector & raster visualizations
├── paper/                        # IEEE-style LaTeX manuscript & publication metadata
└── archive/                      # Deprecated & synthetic artifacts (quarantined)
```

---

## 3. CORE IMPLEMENTATION AUDIT

### 3.1 Classification Implementations (Track A — Camelyon17-WILDS)
- **Primary Architecture (`src/models/resnet18.py`):**
  - Base: ResNet-18 feature backbone (512-D bottleneck representation).
  - Head: Fully-connected linear projection with `Dropout(p=0.5)`.
  - State: Deterministic evaluation mode by default; stochastic mode activated via `enable_dropout()`.
- **Gating Engines (`src/v2/gating/`):**
  - `adaptive_uncertainty.py`: Tiered early-stopping uncertainty gating. Patches with confidence $c \ge 0.90$ are accepted with 1 forward pass; $c < 0.65$ rejected with 1 forward pass; ambiguous margins $c \in [0.65, 0.90)$ trigger chunked MC passes.
  - `multi_signal.py`: Convex combination of confidence, spatial rotation consistency, and feature-space centroid distance.
- **Classification Baseline Status:**
  - Method C1 (Lightweight Confidence Gating, $\tau = 0.85$ to $0.90$) achieves $F_1 = 0.8310 \pm 0.0321$ and AUROC = $0.9355$.
  - Method C7 (Class-Aware Feature SafeAdapt) achieves $F_1 = 0.8318 \pm 0.0314$ and AUROC = $0.9360$.
  - At matched adaptation coverage, C1 admits statistically significantly fewer errors than C7 ($p = 0.0005$).

### 3.2 Detection Implementations (Track B — MIDOG 2021)
- **Primary Architecture:** Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN), operating on $512 \times 512$ tiles at $0.25\,\mu\text{m/px}$.
- **Bio-SafeAdapt Controller (`src/v2/safeadapt/bio_safeadapt.py`):**
  - `estimate_cellularity()`: Hematoxylin Optical Density ($\text{OD}_H = \text{OD}_R - 0.5 \times \text{OD}_B$) projection estimating nuclear cellularity.
  - `monitor_detection_step()`: Cellularity-conditioned silence monitoring. Suppresses false alarms on acellular stroma, adipocytes, and lumens.
  - `apply_elastic_source_anchor()`: Soft parameter regularization ($\lambda_{\text{anchor}} = 0.01$) preventing weights from wandering off the source manifold.
  - `reset_case_boundary()`: Episodic reset restoring certified source weights and flushing optimizer momentum at patient/slide boundaries.

### 3.3 Dataset Loaders & Preprocessing
- **Parquet Loader (`src/data/parquet_camelyon17.py`):** Efficient PyArrow reader streaming $96 \times 96 \times 3$ patches directly from local Parquet shards (`C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data\`).
- **WILDS Interface (`src/data/camelyon17.py`):** Standardizes ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) and patient cluster extraction.
- **MIDOG Loader (`src/data/` & `notebooks/official/kaggle_midog_patches.py`):** Grid window extraction with overlap handling, boundary filtering ($h \ge 16$ px margin), and coordinate space mapping.

---

## 4. CHECKPOINT & CENTROID INTEGRITY AUDIT

All 5 classification source models and per-seed 512-D prototype centroids were inspected in `checkpoints/`:

| Checkpoint File | Architecture | Size | Expected Checkpoint Hash (SHA-256) | Intended Scientific Role |
| :--- | :---: | :---: | :--- | :--- |
| `camelyon17_resnet18_source_s42_BEST.pt` | ResNet-18 | $45,839,231$ bytes | `f0b85ec1a89707a8...` | Source Pre-trained Weights (Seed 42) |
| `camelyon17_resnet18_source_s123_BEST.pt`| ResNet-18 | $45,841,153$ bytes | `cb57d805869f1176...` | Source Pre-trained Weights (Seed 123) |
| `camelyon17_resnet18_source_s2024_BEST.pt`| ResNet-18 | $45,840,579$ bytes | `d9bed5d9bc05edd7...` | Source Pre-trained Weights (Seed 2024) |
| `camelyon17_resnet18_source_s7_BEST.pt`   | ResNet-18 | $45,840,509$ bytes | `9c18d65ab0a97527...` | Source Pre-trained Weights (Seed 7) |
| `camelyon17_resnet18_source_s99_BEST.pt`  | ResNet-18 | $45,840,319$ bytes | `0b299932abd18419...` | Source Pre-trained Weights (Seed 99) |
| `source_centroids_resnet18_s42.pt` | 512-D Tensor | $6,339$ bytes | `a86e80e58d409056...` | Class Prototypes for Seed 42 |
| `source_centroids_resnet18_s123.pt` | 512-D Tensor | $6,346$ bytes | `39e24c98e2cdea80...` | Class Prototypes for Seed 123 |
| `source_centroids_resnet18_s2024.pt` | 512-D Tensor | $6,417$ bytes | `ab3e1190b558c95e...` | Class Prototypes for Seed 2024 |
| `source_centroids_resnet18_s7.pt` | 512-D Tensor | $6,332$ bytes | `71f488b012038366...` | Class Prototypes for Seed 7 |
| `source_centroids_resnet18_s99.pt` | 512-D Tensor | $6,339$ bytes | `3dbaf9bafd6f8258...` | Class Prototypes for Seed 99 |
| `midog_resnet18_s42_BEST.pt` | ResNet-18 | $44,785,291$ bytes | `409a9c7d3b9d1052...` | MIDOG Source Detection Backbone |

---

## 5. ARCHIVED INVALID ARTIFACTS VERIFICATION

The synthetic experiment directory `archive/synthetic_v2_unverified/` was audited:
- Contains: `ARCHIVE_NOTICE.md`, `ABLATION_ANALYSIS.csv`, `UTTA_MED_V2_RESULTS.csv`, and `benchmark_metrics.json`.
- Provenance: Early local smoke test that added Gaussian noise to constants without GPU execution.
- Safeguard: Quarantined outside production paths. `scripts/kaggle/run_kaggle.py` and evaluation pipelines strictly reject synthetic data generators.

---

## 6. KEY REPOSITORY FILE MANIFEST & ROLES

| Subsystem | Key Files | Functional Role | Status |
| :--- | :--- | :--- | :---: |
| **Gating** | `src/v2/gating/adaptive_uncertainty.py`<br>`src/v2/gating/multi_signal.py` | Implementation of adaptive early-stopping & multi-signal confidence | AUDITED |
| **SafeAdapt** | `src/v2/safeadapt/bio_safeadapt.py`<br>`src/v2/safeadapt/controller.py`<br>`src/v2/safeadapt/losses.py` | Implementation of Bio-SafeAdapt, H-OD cellularity, anchoring, and atomic rollbacks | AUDITED |
| **Models** | `src/models/resnet18.py` | Architecture containing head dropout for Monte Carlo uncertainty | AUDITED |
| **Kaggle Automation** | `scripts/kaggle/kaggle_orchestrator.py`<br>`scripts/kaggle/run_kaggle.py`<br>`scripts/kaggle/kernel-metadata.json` | Remote job submission, monitoring, and log retrieval for Dual Tesla T4 | AUDITED |
| **Evaluation** | `scripts/generate_bio_safeadapt_verified_predictions.py`<br>`scripts/generate_bio_safeadapt_ablations.py`<br>`scripts/audit_h3_2_r3b_independent_reproducibility.py` | Independent 1:1 Euclidean greedy matching and 10k case bootstrap | AUDITED |
| **Results** | `FULL_COHORT_MATCHED_COVERAGE.csv`<br>`BIO_SAFEADAPT_VERIFIED_PREDICTIONS.csv`<br>`BIO_SAFEADAPT_ABLATIONS.csv`<br>`MULTI_SCANNER_EVALUATION.csv` | Canonical audited prediction matrices and benchmark results | AUDITED |

*Full 874-entry file manifest with byte sizes and SHA-256 hashes is serialized in `repo_file_manifest.json`.*
