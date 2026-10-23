# UTTA-Med 2.0: Project Audit Report (Phase 1)

**Date:** 2026-10-22  
**Auditor:** Senior AI Research Scientist / Medical Image Analysis Engineer  
**Scope:** Repository architecture, model checkpoints, data pipelines, test coverage, and baseline reproducibility for Project 2 (`UTTA_MED_V1_FINAL`) with reference to Project 1 (`UTTA-Med`).

---

## 1. Executive Summary & Inventory

Project 2 (`UTTA_MED_V1_FINAL`) represents the primary production and publication codebase for **Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology (UTTA-Med)**.

| Metric | Project 2 (Active Dev) | Project 1 (Reference) | Status |
| :--- | :---: | :---: | :--- |
| **Total Non-Cache Files** | 984 files | 999 files | Complete coverage verified |
| **Trained Classification Checkpoints** | 9 ResNet-18 weights (390.48 MB) | 5 ResNet-18 weights (218.6 MB) | Project 2 contains full multi-domain models |
| **Trained Detection Checkpoints** | 3 Faster R-CNN weights (474.15 MB) | 3 Faster R-CNN weights (474.15 MB) | Identical SHA-256 hashes |
| **WSI Bootstrap Trajectory Arrays** | 35 `.npz` files (20.93 MB) | 35 `.npz` files (20.93 MB) | Identical empirical streams |
| **Automated Test Suite** | 120 passed, 0 failed (18.34s) | 114 passed | 100% test pass rate in Project 2 |
| **Publication Draft** | IEEE 8.0-page verified PDF | Draft PDF | Certified US Letter layout |
| **Kaggle CLI Integration** | Authenticated (`~/.kaggle/kaggle.json`) | Unlinked | Ready for GPU remote orchestration |

---

## 2. Directory Layout & Key Components

```
UTTA_MED_V1_FINAL/
├── checkpoints/              # Certified model checkpoints (ResNet-18 Camelyon17 s42/7/99/123/2024, MIDOG, NCT-CRC, PathMNIST)
├── configs/                  # Experiment configuration YAMLs (gating thresholds, learning rates, loss weights)
├── docs/                     # Research manuals, protocol audits, H4/H5 reproducibility dossiers
├── figures/                  # High-resolution Grad-CAM heatmaps, UMAP cluster projections, ROC/PR curves
├── notebooks/                # Official and legacy Kaggle/Colab execution notebooks
├── paper/                    # IEEE manuscript LaTeX sources, certified 8.0-page PDF, submission packages
├── results/                  # Raw and tabulated experimental artifacts
│   ├── data/wsi_bootstrap/   # 35 empirical .npz streaming trajectory arrays across 5 seeds
│   ├── generalization/h3/    # MIDOG scanner generalization, audit suites, and Faster R-CNN models
│   ├── generalization/h4/    # H4 protocol audits, sampling frames, patch manifests
│   └── tables/               # Formatted LaTeX and markdown tables
├── scripts/                  # Seed runners, table verifiers, paper builders, Kaggle jobs
├── src/                      # Production Python package
│   ├── data/                 # Camelyon17-WILDS, MIDOG, NCT-CRC loaders and transforms
│   ├── evaluation/           # Metrics, calibration (ECE, Brier), tent collapse monitors
│   ├── models/               # ResNet-18, Faster R-CNN, MC Dropout wrappers
│   ├── tta/                  # Tent, EATA, SAR, DL-TTA implementations
│   └── v2/                   # SafeAdapt controller, selective gating policies, runtime monitors
└── tests/                    # 120 unit and integration tests verifying guarantees
```

---

## 3. Scientific Verification Matrix

### 3.1 Classification Baselines (Camelyon17-WILDS)
The benchmark evaluates 5 Dutch hospital domains: Centers 0, 3, 4 (train), Center 1 (dev/calibration), Center 2 (unseen target test, 10 slides, 1,280 patches).

| Model / Adaptation Arm | Macro AUROC | Pooled F1 | Harmful Adaptation Rate | Latency (s) | Gating Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frozen Source (ResNet-18)** | 0.9300 | 0.7840 | 0.00% (Baseline) | 168.73s | None (Frozen parameters) |
| **Unconstrained Tent** | 0.6948 | 0.5062 | 18.7% ± 17.9% | 192.10s | None (Entropy minimization on all tokens) |
| **EATA** | 0.9214 | 0.8091 | 2.1% ± 0.8% | 210.45s | Fisher regularized + static entropy filter |
| **Confidence Gating ($\tau=0.90$)** | 0.9362 | 0.8312 | 0.65% ± 0.28% | 168.73s | 1 forward pass ($C(x) \ge 0.90$) |
| **UTTA-Med (MC-Dropout 20-pass)** | 0.9351 | 0.8301 | 0.68% ± 0.31% | 598.21s | 20 stochastic passes ($\mathrm{Var} \le \gamma$) |

*Key Takeaway:* Softmax confidence gating matches MC-Dropout accuracy while eliminating a **3.55× latency penalty** (168.73s vs 598.21s). This empirical finding validates the need for **Adaptive Uncertainty Estimation** in UTTA-Med 2.0.

### 3.2 Dense Detection Baselines (MIDOG 2021)
Evaluates mitotic figure detection from Scanners 1 & 2 to held-out Scanner 3 across 50 cases and 1,100 spatially stratified patches.

| Method | Detection F1 | True Positives | False Positives | Rollback Events | Permanent Fallback |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frozen Source (Faster R-CNN)** | **0.4774** | 243 | 266 | 0 | None |
| **Conventional TTA (Tent-like)** | 0.4702 | 229 | 258 | N/A | Diverged |
| **Monitored TTA** | 0.4709 | 231 | 259 | 4 | No fallback |
| **SafeAdapt (Original)** | **0.4774** | 243 | 266 | 12 | Early fallback triggered |

*Key Takeaway:* Original SafeAdapt successfully protects the system from degradation by falling back to frozen source parameters, but fails to achieve **positive adaptation transfer**. The improved **Bio-SafeAdapt** controller must use cellularity conditioning and continuous parameter anchoring to achieve positive adaptation ($F_1 > 0.4774$).

---

## 4. Immediate Research Directives for Project 2 (UTTA-Med 2.0)

1. **Adaptive Uncertainty Estimation (`src/v2/gating/adaptive_uncertainty.py`)**:
   - Single-pass confidence filter first.
   - Dynamic stochastic pass budget ($K \in \{5, 10, 20\}$) triggered only when confidence falls in the uncertainty band $[0.65, 0.90)$.
   - Early stopping when epistemic variance stabilizes ($|\sigma_{k}^2 - \sigma_{k-5}^2| < 10^{-4}$).

2. **Multi-Signal Gating (`src/v2/gating/multi_signal.py`)**:
   - Combine MC-Dropout variance ($U_{\mathrm{MC}}$), test-time augmentation consistency ($U_{\mathrm{consistency}}$ via 90° rotation and H-E color perturbation), and source centroid distance ($U_{\mathrm{feature}}$).
   - $U_{\mathrm{total}} = \alpha U_{\mathrm{MC}} + \beta U_{\mathrm{consistency}} + \gamma U_{\mathrm{feature}}$.

3. **Bio-SafeAdapt Detection Controller (`src/v2/safeadapt/bio_safeadapt.py`)**:
   - Cellularity-conditioned silence monitoring via hematoxylin optical density (H-OD) proxy.
   - Continuous elastic source anchoring ($L_{\mathrm{anchor}} = \lambda \|\theta - \theta_0\|_2^2$).
   - Episodic patient-level resets at case boundaries.

4. **Kaggle GPU Orchestration Suite (`scripts/kaggle/`)**:
   - Automated kernel push, monitoring, log polling, and artifact retrieval via the authenticated Kaggle API.
