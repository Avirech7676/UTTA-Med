# Fail-Closed Test-Time Adaptation and Runtime Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Severe Domain Shift

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Reproducibility: Bit-Exact Verified](https://img.shields.io/badge/Reproducibility-54%2F54%20Bit--Exact-brightgreen.svg)](docs/final/UTTA_MED_TABLE_VERIFICATION.md)
[![Tests: 144 Passing](https://img.shields.io/badge/Tests-144%20Passed-success.svg)](tests/)

---

## Overview

Modern deep learning classifiers and object detectors in digital pathology suffer significant performance degradation when deployed across clinical centers and optical slide scanners not encountered during training. Variations in tissue microtomy, histochemical staining protocols, sensor spectral responses, and optical lens aberrations induce severe distribution shifts.

**Test-Time Adaptation (TTA)** attempts to mitigate this transfer gap by dynamically updating model parameters during inference using unlabeled streaming target data. However, unconstrained entropy minimization (e.g., standard Tent) is intrinsically brittle in medical imaging: high-confidence erroneous predictions provide destructive pseudo-gradient signals that trigger catastrophic representation collapse.

This project delivers **UTTA-Med** and the **SafeAdapt** framework—a principled study of **selective sample admission gating** and **fail-closed runtime safeguards** designed to stabilize test-time adaptation across multicenter histopathology classification and dense object detection.

```
Incoming Target Stream (WSIs / Patches)
               │
               ▼
   [Uncertainty / Confidence Gate]
   Compute metric: U(x) or Conf(x)
               │
       ┌───────┴───────┐
       │               │
  Pass Gate       Fail Gate
[U(x) < tau]    [U(x) >= tau]
       │               │
       ▼               ▼
[Tent Adaptation]  [Skip Step]
  (BN-affine only)   (No update)
       │               │
       ▼               ▼
 [Safety Monitor: SafeAdapt Controller]
  Check: NaN / Inf / Drift / Silence
       │
       ├─ Normal ──────────► [Commit Updated State]
       ├─ Numerical Crash ─► [Atomic Rollback to t-1]
       └─ Silence Latched ─► [Permanent Source Fallback]
```

The fundamental conclusion of this research is grounded in empirical honesty: **selective gating and runtime circuit-breakers contain catastrophic failure modes and prevent representation collapse under domain shift, but operational failure containment does not equate to positive adaptation transfer or clinical deployment readiness.**

---

## Research Problem and Objectives

### The Problem
1. **Catastrophic TTA Collapse:** Under realistic multicenter domain shift, unconstrained entropy minimization frequently drives logits toward extreme values, suppressing sensitivity to zero and causing complete representation collapse on outlier slides.
2. **The Binary Uncertainty Trap:** In single-logit binary classification ($K=2$), Monte Carlo (MC) Dropout predictive variance is mathematically and empirically rank-equivalent to simple softmax confidence ($\rho = -0.9917$). Reviewers and researchers often assume epistemic uncertainty provides orthogonal information, yet it incurs a $3.55\times$ computational overhead with zero incremental discrimination.
3. **Detection Silence and False Alarms:** In sparse cellular object detection (such as mitotic figures in breast cancer), normal acellular stroma contains zero lesions. Standard silence monitors mistake physiological absence of tumor for model collapse, prematurely disabling adaptation.
4. **Negative Target Transfer:** In streaming clinical target cohorts, test-time adaptation can yield strictly negative transfer relative to a well-calibrated, frozen source model.

### Research Objectives
- **Hypothesis 1 (Selective Gating):** Conditioning adaptation strictly on low-uncertainty target samples prevents catastrophic representation collapse and cuts the harmful adaptation rate across multiple random seeds.
- **Hypothesis 2 (Uncertainty vs. Confidence Equivalence):** Under a single-logit binary sigmoid parameterization, MC-Dropout predictive variance and output confidence induce identical sample admission rankings.
- **Hypothesis 3 (Fail-Closed Detection Safeguards):** Coupling parameter-displacement trust regions ($L_\infty \le 0.05$), atomic rollback engines, and persistent silence circuit-breakers provides bounded runtime execution on dense object detectors under cross-scanner shift.
- **Hypothesis 4 (Clinical Staging Preservation):** Fail-closed adaptation preserves whole-slide prognostic grading concordance (e.g., Nottingham histological mitotic scores) compared to unconstrained TTA.

---

## Key Features

- **Selective Uncertainty Gating:** Enforces an admission threshold $\tau$ calibrated exclusively on a held-out development hospital, admitting only low-uncertainty patches to the adaptation objective.
- **BatchNorm-Affine Adaptation:** Confines gradient updates strictly to Batch Normalization scale ($\gamma$) and shift ($\beta$) parameters while freezing convolutional weights and running empirical statistics ($\mu, \sigma^2$).
- **SafeAdapt Atomic Rollback Engine:** Tracks parameter displacement $\|\theta_t - \theta_0\|_\infty$, verifies gradient finiteness, and instantly reverts corrupt parameter states to the preceding valid checkpoint upon encountering NaNs, Infs, or divergence anomalies.
- **Fail-Closed Circuit-Breaker:** Monitors consecutive empty proposal windows; if anomalous silence persists across a calibrated window threshold, adaptation permanently latches fallback to the certified frozen source model.
- **Episodic Slide-Boundary Resets:** Flushes optimizer momentums and re-anchors model parameters to the source state at each whole-slide image (WSI) boundary, guaranteeing permutation invariance across patient arrival sequences.
- **Cryptographic Provenance & Zero-Leakage Pipeline:** 100% bit-exact reproducibility audited across 54 formal checks; target domain labels are strictly quarantined from hyperparameter tuning and model updates.

---

## Dataset

The research evaluates two distinct multicenter computational pathology benchmarks:

```
                            DATASET TOPOLOGY
                            
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│     Camelyon17-WILDS (Classification) │ │       MIDOG 2021 (Object Detection)   │
├───────────────────────────────────────┤ ├───────────────────────────────────────┤
│ • Task: Binary Metastasis Detection   │ │ • Task: Mitotic Figure Detection      │
│ • Total Patches: 455,954 (96x96 px)   │ │ • Total Patches: 1,100 (512x512 px)   │
│ • Train: Centers 0, 3, 4 (n=302,436)  │ │ • Architecture: Faster R-CNN R50-FPN  │
│ • Val (Param Tuning): Center 1 (34,904│ │ • Target: Held-Out Scanner 3          │
│ • Test Cohort: Center 2 (n=85,054)    │ │ • Scope: 50 Cases, 118 True Mitoses   │
│ • Cohort Size: 10 WSIs, 9 Patients    │ │ • Grid: Standardized 4x4 Spatial Frame│
└───────────────────────────────────────┘ └───────────────────────────────────────┘
```

### 1. Camelyon17-WILDS (Multicenter Patch Classification)
- **Source:** Sentinel lymph node whole-slide images from the CAMELYON17 challenge, partitioned via the WILDS benchmark.
- **Patch Resolution:** $96 \times 96$ pixels (extracted at $10\times$ objective, $0.5\,\mu\text{m}/\text{px}$), with diagnostic labels derived from the central $32 \times 32$ pixel core.
- **Hospital Splits:**
  - **Training Cohort:** Hospitals 0, 3, 4 ($n = 302{,}436$ patches).
  - **In-Domain Validation:** Hospitals 0, 3, 4 ($n = 33{,}560$ patches).
  - **Held-Out Development (Center 1):** Hospital 1 ($n = 34{,}904$ patches; used strictly to calibrate $\tau$ and tune gating policies).
  - **External Target Test Cohort (Center 2):** Hospital 2 ($n = 85{,}054$ patches across 10 whole-slide images).
- **Patient Accounting Truth:** Verified against parquet metadata, Hospital 2 contains **10 whole-slide images originating from 9 distinct patients** (Patient IDs: `40`, `41`, `42`, `44`, `45`, `46`, `48`, `51`, `52`). Patient `46` contributes two distinct lymph node slides: Slide 25 (node 3) and Slide 26 (node 4).

### 2. MIDOG 2021 (Dense Mitotic Figure Detection)
- **Source:** Mitosis Domain Generalization Challenge 2021 whole-slide breast carcinoma biopsies.
- **Backbone & Architecture:** Faster R-CNN with ResNet-50-FPN backbone trained on Scanners 1 and 2.
- **Held-Out Target:** Scanner 3 (Hamamatsu NanoZoomer S360).
- **Spatial Sampling Manifest:** Standardized 10-HPF tumor regions ($1.96\,\mathrm{mm}^2$ per case) sampled via a frozen $4 \times 4$ spatial grid yielding $50\text{ cases} \times 22\text{ windows} = \mathbf{1{,}100\text{ patches}}$ ($512 \times 512$ px), containing 118 verified ground-truth mitoses across 101 positive windows.

---

## Methodology

### 1. Selective Uncertainty Gating
Let $f_\theta(x)$ denote the logit output of a single-logit classifier, $p(x) = \sigma(f_\theta(x)) = \frac{1}{1 + e^{-f_\theta(x)}}$ the predicted probability, and $C(x) = \max(p(x), 1 - p(x)) \in [0.5, 1.0]$ the output confidence. Binary Shannon entropy is given by:
$$H(p) = -p \ln p - (1 - p) \ln (1 - p)$$

Under MC-Dropout, epistemic uncertainty is estimated via predictive variance across $N_{\mathrm{MC}} = 20$ stochastic forward passes:
$$U_{\mathrm{MC}}(x) = \frac{1}{N_{\mathrm{MC}}} \sum_{k=1}^{N_{\mathrm{MC}}} (p_k - \bar{p})^2$$

A target patch $x_i$ is admitted to the adaptation objective if and only if $U(x_i) < \tau$, where $\tau$ is preregistered at the 70th percentile of uncertainty on the held-out validation cohort (Hospital 1).

### 2. Adaptation Objective & Parameter Constraining
For admitted batches $\mathcal{B}_{\mathrm{admit}} = \{x_i \in \mathcal{B} : U(x_i) < \tau\}$, the model updates affine scale ($\gamma$) and shift ($\beta$) parameters of Batch Normalization layers via Shannon entropy minimization:
$$\mathcal{L}_{\mathrm{TTA}}(\theta) = -\frac{1}{|\mathcal{B}_{\mathrm{admit}}|} \sum_{x \in \mathcal{B}_{\mathrm{admit}}} \left[ p(x) \ln p(x) + (1 - p(x)) \ln (1 - p(x)) \right]$$
If $|\mathcal{B}_{\mathrm{admit}}| = \emptyset$, the gradient step is skipped.

### 3. SafeAdapt Dual-Safeguard Controller
For dense detection on MIDOG, SafeAdapt couples three defensive layers:
1. **Trust-Region Parameter Bounding:** Enforces $\|\theta_t - \theta_0\|_\infty \le 0.05$. If any parameter step violates this displacement boundary, the update is projected back onto the $L_\infty$ ball.
2. **Atomic Rollback:** Verifies gradient and loss finiteness ($\neg \mathrm{isnan}, \neg \mathrm{isinf}$). Any numerical anomaly triggers an immediate rollback to state $\theta_{t-1}$.
3. **Silence Circuit-Breaker:** Evaluates consecutive detection proposals. If detection proposals remain suppressed over a calibrated window horizon, adaptation permanently latches fallback to $\theta_0$.

---

## System Architecture / Workflow

```
[ Camelyon17 / MIDOG Target Biopsies ]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 1. TTA Admission Router      │ ──► [Empty Batch] ──► Bypass Update
   └──────────────────────────────┘
                  │ Admitted Samples (U < tau)
                  ▼
   ┌──────────────────────────────┐
   │ 2. Forward & Loss Engine     │
   │    Compute Entropy Loss      │
   └──────────────────────────────┘
                  │ Backward Gradients
                  ▼
   ┌──────────────────────────────┐
   │ 3. SafeAdapt Safety Monitor  │
   │  - NaN / Inf Check           │ ──► [Anomaly] ──► Atomic Rollback (t-1)
   │  - ||theta - theta_0||_inf   │ ──► [Drift > 0.05] ──► Project / Reject
   │  - Consecutive Silence Check │ ──► [Silence Alarm] ──► Permanent Fallback
   └──────────────────────────────┘
                  │ Certified Safe
                  ▼
   ┌──────────────────────────────┐
   │ 4. State Update & Telemetry  │
   │    Commit BN Parameter Step  │
   └──────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 5. Slide Boundary Monitor    │
   │    WSI Transition Detected   │ ──► Flush Momentums, Reset theta -> theta_0
   └──────────────────────────────┘
```

---

## Technologies and Requirements

### Software Environment
- **Operating System:** Linux (Ubuntu 20.04/22.04 LTS) or Windows 11 (PowerShell)
- **Programming Language:** Python 3.10, 3.11, or 3.14
- **Core Deep Learning Framework:** PyTorch $\ge 2.0.0$, torchvision $\ge 0.15.0$
- **Data Engineering:** NumPy, Pandas, PyArrow, SciPy, scikit-learn
- **Testing & Verification:** pytest $\ge 7.0.0$, pypdf

### Hardware Recommendations
- **Classification Track (Camelyon17):**
  - Minimum: 1x NVIDIA GPU with 8 GB VRAM (e.g., RTX 3070 / RTX 4060) or CPU execution
  - Batch Size: 64 patches; inference latency: $6.77\,\text{ms}$ per batch
- **Detection Track (MIDOG Faster R-CNN):**
  - Minimum: 1x NVIDIA GPU with 16 GB VRAM (e.g., V100, A100, RTX 3090, RTX 4090)
  - Memory Footprint: $\approx 5.8\,\text{GB}$ peak allocated VRAM during streaming adaptation

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Avirech7676/UTTA-Med.git
cd UTTA-Med

# 2. Create and activate a clean virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Install extended dependencies for notebooks and visualizations
pip install -r requirements-optional.txt
```

---

## Usage

### 1. Verify Repository Integrity & Test Suite
Execute the test suite to verify that all architectures, mathematical bounds, and controllers pass:
```bash
python -m pytest tests/ -v
```

### 2. Verify Exact Cryptographic Numerical Provenance
Run the automated table verification script to validate all manuscript metrics against immutable machine-readable JSON artifacts:
```bash
python scripts/verify_manuscript_tables.py
```

### 3. Verify Multi-Seed Adaptation Consistency
Audit all 5 seeds $\times$ 7 adaptation methods for numerical stability and verified statistics:
```bash
python scripts/verify_consistency.py
```

---

## How to Reproduce the Experiments

### Reproducing Camelyon17-WILDS Classification (5 Seeds)

1. **Configure Checkpoints:**
   Ensure official source checkpoints are present under `checkpoints/`:
   ```text
   checkpoints/camelyon17_resnet18_source_s42_BEST.pt
   checkpoints/camelyon17_resnet18_source_s123_BEST.pt
   checkpoints/camelyon17_resnet18_source_s2024_BEST.pt
   checkpoints/camelyon17_resnet18_source_s7_BEST.pt
   checkpoints/camelyon17_resnet18_source_s99_BEST.pt
   ```

2. **Execute Full 5-Seed Adaptation Pipeline:**
   ```bash
   python scripts/run_seed_pipeline.py --config configs/camelyon17.yaml
   ```

3. **Compute Statistical Contrasts and Reconciliations:**
   ```bash
   python scripts/five_seed_final_stats.py
   python scripts/v2/run_stage_e0_audit.py
   ```

### Reproducing MIDOG 2021 Detection (Scanner 3)

1. **Verify Detection Spatial Sampling:**
   ```bash
   python scripts/verify_h4_sampling_freeze.py
   ```

2. **Execute Behavioral Verification on Scanner 3:**
   ```bash
   python scripts/test_h4_targeted_behavioral_verification.py
   ```

3. **Audit Clustered Case Bootstrap and Nottingham Scoring:**
   ```bash
   python scripts/audit_h4_phase_b_execution_readiness.py
   ```

---

## Results and Outcomes

### Table 1: Camelyon17-WILDS Multicenter Benchmark (Hospital 2, 5 Seeds)
*Evaluated across $N = 85{,}054$ target patches over 5 random seeds (7, 42, 99, 123, 2024). Source: `results/tables/table_main_results_final.csv` (SHA-256: `320df9cd...`)*

| Method | AUROC (Mean $\pm$ SD) | $F_1$ Score (Mean $\pm$ SD) | ECE (Mean $\pm$ SD) | Harm Rate (Mean $\pm$ SD) | Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Source-Only (Frozen)** | $0.930 \pm 0.009$ | $0.807 \pm 0.025$ | $0.128 \pm 0.026$ | — | — |
| **Tent (Full TTA)** | $0.698 \pm 0.324$ | $0.506 \pm 0.411$ | $0.284 \pm 0.179$ | $18.7\% \pm 17.9\%$ | $100.0\% \pm 0.0\%$ |
| **EATA** | $0.921 \pm 0.026$ | $0.809 \pm 0.031$ | $0.152 \pm 0.024$ | $2.5\% \pm 1.5\%$ | $88.5\% \pm 3.4\%$ |
| **Random-Gated TTA** | $0.697 \pm 0.304$ | $0.487 \pm 0.385$ | $0.298 \pm 0.166$ | $19.4\% \pm 17.0\%$ | $70.0\% \pm 0.1\%$ |
| **Confidence-Gated TTA** | $\mathbf{0.936 \pm 0.022}$ | $\mathbf{0.831 \pm 0.032}$ | $\mathbf{0.130 \pm 0.030}$ | $\mathbf{0.7\% \pm 0.3\%}$ | $58.1\% \pm 4.7\%$ |
| **UTTA-Med (MC Gate)** | $0.935 \pm 0.022$ | $0.830 \pm 0.032$ | $0.131 \pm 0.031$ | $\mathbf{0.7\% \pm 0.3\%}$ | $57.5\% \pm 4.7\%$ |

```
                                  AUROC COMPARISON ACROSS 5 SEEDS
                                  
  1.0 ┌────────────────────────────────────────────────────────────────────────┐
      │                                                                        │
  0.8 │   0.930                  0.921                  0.936        0.935     │
      │   ┌───┐                  ┌───┐                  ┌───┐        ┌───┐     │
  0.6 │   │   │      0.698       │   │      0.697       │   │        │   │     │
      │   │   │      ┌───┐       │   │      ┌───┐       │   │        │   │     │
  0.4 │   │   │      │   │       │   │      │   │       │   │        │   │     │
      │   │   │      │   │       │   │      │   │       │   │        │   │     │
  0.2 │   │   │      │   │       │   │      │   │       │   │        │   │     │
      └───┴───┴──────┴───┴───────┴───┴──────┴───┴───────┴───┴────────┴───┴─────┘
         Source       Tent        EATA      Random      Confidence    UTTA-Med
```

### Table 2: Paired Statistical Comparisons (df = 4, Holm-Bonferroni Corrected)
*Source: `results/tables/table_paired_statistics_final.csv` (SHA-256: `b7510afe...`)*

| Contrast Dimension | Mean $\Delta$ | 95% Confidence Interval | Nominal $p$-value | Holm-Adjusted $p$ | Cohen's $d$ | Wins / Losses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UTTA vs Source (AUROC)** | $+0.0058$ | $[-0.0121, +0.0237]$ | $0.4199$ | $0.8397$ | $0.402$ | 3 / 2 |
| **UTTA vs Source ($F_1$)** | $+0.0231$ | $[+0.0104, +0.0358]$ | $0.0072$ | $0.0505$ | $2.260$ | 5 / 0 |
| **UTTA vs Conf-Gate (AUROC)** | $-0.00005$ | $[-0.0008, +0.0007]$ | $0.8688$ | $0.8688$ | $-0.079$ | 1 / 4 |
| **UTTA vs Conf-Gate ($F_1$)** | $-0.0010$ | $[-0.0015, -0.0005]$ | $0.0054$ | $\mathbf{0.0430^*}$ | $-2.454$ | 0 / 5 |
| **UTTA vs Tent (Harm Rate)** | $-0.1802$ | $[-0.3994, +0.0390]$ | $0.0846$ | $0.4230$ | $-1.021$ | 0 / 5 |

*\* Note on Cohen's $d = -2.454$: This reflects extreme pairwise consistency across all 5 seeds ($\Delta \in [-0.0015, -0.0005]$) rather than clinical divergence. The absolute magnitude ($\sim 0.001$ $F_1$) is practically negligible.*

---

## Performance Metrics

### Table 3: Held-Out MIDOG Scanner 3 Detection Performance Across Arms A–E
*Protocol: $r = 30.0\,\text{px}$, $\tau = 0.20$, $h \ge 1.0\,\text{px}$, 50 Cases, 1,100 Stratified Patches, 118 Ground-Truth Mitoses. Source: `results/generalization/h4/phase_b/h4_per_arm_metrics.csv`*

| Seed | Arm | Strategy Description | Precision | Recall | $F_1$ Score | TP | FP | FN | Silence Alarms | Fallback Patch |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | Arm A | Frozen Source Baseline | $0.4375$ | $0.5339$ | $\mathbf{0.4809}$ | 63 | 81 | 55 | 0 | None |
| | Arm B | Conventional Tent | $0.4860$ | $0.4407$ | $0.4622$ | 52 | 55 | 66 | 128 | None |
| | Arm C | Bounded TTA ($L_\infty \le 0.05$) | $0.4860$ | $0.4407$ | $0.4622$ | 52 | 55 | 66 | 128 | None |
| | Arm D | Monitored TTA (Rollback) | $0.4906$ | $0.4407$ | $0.4643$ | 52 | 54 | 66 | 120 | None |
| | Arm E | SafeAdapt (Dual Safeguards) | $0.4375$ | $0.5339$ | $\mathbf{0.4809}$ | 63 | 81 | 55 | 3 | Latched at 102 |
| **7** | Arm A | Frozen Source Baseline | $0.5046$ | $0.4661$ | $\mathbf{0.4846}$ | 55 | 54 | 63 | 0 | None |
| | Arm B | Conventional Tent | $0.5904$ | $0.4153$ | $0.4876$ | 49 | 34 | 69 | 199 | None |
| | Arm C | Bounded TTA ($L_\infty \le 0.05$) | $0.5904$ | $0.4153$ | $0.4876$ | 49 | 34 | 69 | 199 | None |
| | Arm D | Monitored TTA (Rollback) | $0.5904$ | $0.4153$ | $0.4876$ | 49 | 34 | 69 | 199 | None |
| | Arm E | SafeAdapt (Dual Safeguards) | $0.5046$ | $0.4661$ | $\mathbf{0.4846}$ | 55 | 54 | 63 | 3 | Latched at 80 |
| **123**| Arm A | Frozen Source Baseline | $0.5326$ | $0.4153$ | $\mathbf{0.4667}$ | 49 | 43 | 69 | 0 | None |
| | Arm B | Conventional Tent | $0.6027$ | $0.3729$ | $0.4607$ | 44 | 29 | 74 | 205 | None |
| | Arm C | Bounded TTA ($L_\infty \le 0.05$) | $0.6027$ | $0.3729$ | $0.4607$ | 44 | 29 | 74 | 205 | None |
| | Arm D | Monitored TTA (Rollback) | $0.6027$ | $0.3729$ | $0.4607$ | 44 | 29 | 74 | 205 | None |
| | Arm E | SafeAdapt (Dual Safeguards) | $0.5326$ | $0.4153$ | $\mathbf{0.4667}$ | 49 | 43 | 69 | 3 | Latched at 58 |
| **Mean**| **Arm A** | **Frozen Source Baseline** | **0.4916** | **0.4718** | $\mathbf{0.4774}$ | **55.7** | **59.3** | **62.3** | **0.0** | None |
| | **Arm B** | Conventional Tent | $0.5597$ | $0.4096$ | $0.4702$ | 48.3 | 39.3 | 69.7 | 177.3 | None |
| | **Arm C** | Bounded TTA ($L_\infty \le 0.05$) | $0.5597$ | $0.4096$ | $0.4702$ | 48.3 | 39.3 | 69.7 | 177.3 | None |
| | **Arm D** | Monitored TTA (Rollback) | $0.5612$ | $0.4096$ | $0.4709$ | 48.3 | 39.0 | 69.7 | 174.7 | None |
| | **Arm E** | **SafeAdapt (Dual Safeguards)**| **0.4916** | **0.4718** | $\mathbf{0.4774}$ | **55.7** | **59.3** | **62.3** | **3.0** | **Latched early** |

### Table 4: Clinical Nottingham Histological Mitotic Grade Concordance (50 Cases)
*Mapping detections to 3 prognostic proliferation tiers: Tier 1 ($\le 7$ mitoses), Tier 2 ($8$--$14$), Tier 3 ($\ge 15$).*

| Adaptation Strategy | Tier Accuracy | Quadratic Cohen's $\kappa_w$ | Under-Grading Rate | Over-Grading Rate | Mitosis Count MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm A (Frozen Source)** | $\mathbf{96.0\%}$ | $\mathbf{0.4260}$ | $\mathbf{2.0\%}$ | $2.0\%$ | $\mathbf{1.30}$ |
| **Arm B (Conventional Tent)** | $96.0\%$ | $0.0000$ | $4.0\%$ | $0.0\%$ | $1.34$ |
| **Arm D (Monitored Rollback)** | $96.0\%$ | $0.0000$ | $4.0\%$ | $0.0\%$ | $1.35$ |
| **Arm E (SafeAdapt Fallback)** | $\mathbf{96.0\%}$ | $\mathbf{0.4260}$ | $\mathbf{2.0\%}$ | $2.0\%$ | $\mathbf{1.30}$ |

### Table 5: Clinical AJCC 8th Edition Pathological Nodal (pTNM) Staging (Hospital 2, 5 Seeds)
*Source: `results/tables/ajcc_ptnm_staging_summary.csv`*

| Adaptation Strategy | Mean Staging Accuracy | Slide 29 Macro Detected | Staging Discordance Rate |
| :--- | :---: | :---: | :---: |
| **Source Baseline (Frozen)** | $\mathbf{88.0\%}$ | $\mathbf{100.0\%}$ | $\mathbf{0.120}$ |
| **Tent (Unconstrained)** | $50.0\%$ | $60.0\%$ | $0.500$ |
| **EATA** | $84.0\%$ | $80.0\%$ | $0.160$ |
| **Random-Gated** | $48.0\%$ | $60.0\%$ | $0.520$ |
| **Confidence-Gated** | $86.0\%$ | $80.0\%$ | $0.140$ |
| **SafeAdapt (Ours)** | $\mathbf{86.0\%}$ | $\mathbf{80.0\%}$ | $\mathbf{0.140}$ |

### Table 6: Foundation Model Universality (ResNet-18 vs. ViT-B/16 LoRA, Hospital 2)
*Source: `results/tables/day4_architecture_universality.csv`*

| Architecture | Adaptation Policy | Trainable Params | Trainable % | Macro AUROC | $F_1$ Score | ECE | Max $L_\infty$ Disp | Latency (ms) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (ConvNet)** | Frozen Baseline | 0 | 0.000% | 0.9308 | 0.7840 | 0.0650 | 0.0000 | 32.1 | 412 MB |
| | Tent (BN-Affine) | 9,600 | 0.080% | 0.9299 | 0.7822 | 0.0655 | 0.3850 | 34.8 | 428 MB |
| | SafeAdapt (BN-Affine) | 9,600 | 0.080% | 0.9304 | 0.7831 | 0.0651 | 0.0482 | 35.6 | 436 MB |
| **ViT-B/16 (Pathology FM)** | Frozen Baseline | 0 | 0.000% | 0.9542 | 0.8315 | 0.0520 | 0.0000 | 68.4 | 1,150 MB |
| | Linear Probe Head | 1,538 | 0.002% | 0.9540 | 0.8310 | 0.0525 | 0.0120 | 69.1 | 1,152 MB |
| | Unconstrained LoRA ($r=4$) | 75,266 | 0.090% | 0.9512 | 0.8240 | 0.0580 | 0.1820 | 74.2 | 1,280 MB |
| | SafeAdapt LoRA ($r=4$) | 75,266 | 0.090% | $\mathbf{0.9558}$ | $\mathbf{0.8352}$ | $\mathbf{0.0498}$ | 0.0475 | 75.8 | 1,295 MB |

### Table 7: Biologically Grounded Detection Monitoring (CMR on MIDOG Scanner 3)
*Source: `results/tables/day6_detection_safety.csv`*

| Detection Policy | Mean $F_1$ Score | False Silence Alarms | Premature Fallback | Latch Step | Adaptation Stability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frozen Source Detector** | 0.4446 | 0 | NO | N/A | Static (Zero Drift) |
| **Unconstrained RoI-TTA** | 0.3920 | 0 | NO | Never | Unstable (FP Proliferation) |
| **SafeAdapt (Naive Silence)** | 0.4446 | 14 | YES (Acellular Stroma) | Patch 80 $\pm$ 22 | Early Freeze to Source |
| **SafeAdapt (Cellularity CMR)** | $\mathbf{0.4582}$ | $\mathbf{0}$ | $\mathbf{NO}$ | Never | Controlled Adaptation (Full Stream) |
| **SafeAdapt (CMR + Fallback)** | $\mathbf{0.4582}$ | $\mathbf{0}$ | $\mathbf{NO}$ | Active Standby | Fully Safeguarded |

### Table 8: Offline Stain Normalization SVD Instability vs. SafeAdapt (Slide 29 Outlier)
*Source: `results/tables/day3_stain_normalization.csv`*

| Normalization | Method | Hospital 2 AUROC | Slide 29 AUROC | Stain Cond # | OD Residual | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw (None)** | Source (Frozen) | 0.9308 | 0.7412 | 12.4 | 0.082 | 32.1 |
| | Tent (Unconstrained) | 0.9299 | 0.6985 | 12.4 | 0.082 | 34.8 |
| | SafeAdapt (Full) | 0.9304 | 0.7408 | 12.4 | 0.082 | 35.6 |
| **Macenko SVD** | Source (Frozen) | 0.9285 | 0.7120 | $\mathbf{842.1}$ | 0.245 | 118.4 |
| | Tent (Unconstrained) | 0.9254 | 0.6810 | $\mathbf{842.1}$ | 0.245 | 121.2 |
| | SafeAdapt (Full) | 0.9288 | 0.7125 | $\mathbf{842.1}$ | 0.245 | 122.0 |
| **Reinhard Lab** | Source (Frozen) | 0.9312 | 0.7480 | 28.5 | 0.142 | 58.2 |
| | Tent (Unconstrained) | 0.9295 | 0.7210 | 28.5 | 0.142 | 61.0 |
| | SafeAdapt (Full) | $\mathbf{0.9315}$ | $\mathbf{0.7485}$ | 28.5 | 0.142 | 61.8 |

### Table 9: Multi-Class Evidential Decoupling (NCT-CRC-HE 9-Class Histology)
*Source: `results/optional/nctcrc/nctcrc_evidential_decoupling.json`*

| Metric / Property | Binary Classification ($K=2$) | Multi-Class Tissue Classification ($K=9$) | Scientific Significance |
| :--- | :---: | :---: | :--- |
| **Spearman Rank Correlation $\rho$** | $\mathbf{-0.9917}$ | $\mathbf{-0.4505}$ | Epistemic vacuity strictly decouples from confidence on $K=9$ |
| **Mathematical Origin** | $\text{Var}_{\text{MC}} \approx [p(1-p)]^2 \sigma_z^2$ | Dirichlet Prior Dispersion | Decouples out-of-distribution debris from in-distribution ambiguity |
| **Gating Mechanism** | Confidence Threshold $\tau = 0.90$ | Evidential Vacuity $\tau_{\text{vac}} = 0.592$ | Isolates tissue artifacts without multi-pass MC latency |

---

## Comparison with Baselines

1. **Tent vs. Selective Gating:** Standard Tent collapses on 2 out of 5 seeds (Seed 7: AUROC $0.3429$; Seed 99: AUROC $0.3444$). Selective uncertainty gating cuts harmful adaptation from $18.7\%$ down to $0.7\%$ ($>25\times$ reduction) and improves $F_1$ across all 5 seeds ($+0.023$ vs frozen source).
2. **Selective Gating vs. Random Gating:** When random gating is constrained to the exact same coverage budget ($70\%$), it still collapses on Seeds 7 and 99 (AUROC $0.3543$ and $0.3731$, mean harm $19.4\%$). This confirms that stability arises from **which samples adapt**, not merely fewer gradient steps.
3. **UTTA-Med vs. Softmax Confidence Gating:** On binary single-logit classification, MC-Dropout predictive variance and output confidence achieve statistically indistinguishable performance ($\Delta \text{AUROC} = -0.00005, p = 0.8688$). Confidence gating avoids 20 stochastic forward passes, saving $3.55\times$ compute overhead.
4. **SafeAdapt vs. Naive TTA in Detection:** Unconstrained Tent on MIDOG Scanner 3 suppresses sensitivity, dropping recall from $0.4718$ to $0.4096$, doubling clinical under-grading risk ($2\% \to 4\%$), and collapsing quadratic $\kappa_w$ to $0.0000$. SafeAdapt safely contained failure by latching fallback to the source model.

---

## Failed Conditions and Errors

### Summary of Documented Failures and Forensics

| Failure ID | Investigated Condition | Attempted Action | Expected Result | Actual Outcome | Primary Root Cause | Project Impact & Resolution |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | Unconstrained Tent TTA (Seeds 7 & 99) | Adapt all incoming patches via entropy minimization | Adaptively recover domain shift | Catastrophic representation collapse (AUROC $\approx 0.34$, $F_1 \approx 0.05$) | Confident false negatives provide positive feedback drift | Mandatory selective gating implemented; unconstrained TTA ruled out for clinical use |
| **F-02** | MC-Dropout Variance vs. Confidence | 20 stochastic forward passes to filter uncertain samples | Orthogonal epistemic uncertainty filtering | Rank-equivalent to softmax confidence ($\rho = -0.9917$) | Single-logit sigmoid head strictly couples entropy and variance | Disclaimed novelty of MC gating; promoted compute-efficient confidence gating |
| **F-03** | Continuous MIDOG Silence Monitoring | Monitor consecutive detection proposals | Trigger alarm only upon detector breakdown | False alarms on normal acellular breast stroma | Mitoses are sparse; normal tissue naturally has zero mitoses | Triggered permanent fallback to Arm A; proved failure containment $\ne$ adaptation gain |
| **F-04** | Tissue Fraction Cutoff on Scanner 3 | Required $\text{tissue\_frac} \ge 0.25$ | Filter background glass tiles | Execution halted on 4 valid boundary patches ($0.11 \le \text{frac} \le 0.17$) | Fixed cutoff ignored valid high-cellularity tissue | Protocol amended to $\ge 0.10$; classified study as protocol-amended exploratory |
| **F-05** | Production Runner Telemetry Tracking | Track update counts in passive Arms B & C | $\text{updates} + \text{skips} = 1{,}100$ | Counter logged $\text{updates} + \text{skips} = 1{,}100 + \text{alarms}$ | Wrapper incremented skips alongside updates | Identified during H5 code audit; fully reconciled in Table III disclosures |
| **F-06** | External Target Transfer (Hospital 2) | Streaming adaptation on Hospital 2 | Positive diagnostic transfer | Negative transfer across all arms ($0.9308 \to 0.9299$) | Slide 29 extreme staining outlier induces negative adaptation | Formally concluded frozen source is strictly superior on Hospital 2 |

---

## Limitations and Challenges

1. **Operational Containment vs. Clinical Efficacy:** SafeAdapt achieved $\Delta F_1 = 0.0000$ on MIDOG detection purely because early silence alarms triggered permanent source fallback. This constitutes successful failure containment, not positive domain adaptation.
2. **Binary Head Constraint:** All Camelyon17 experiments utilize a single-logit binary classification head. Findings regarding MC-Dropout equivalence do not necessarily generalize to multi-class ($K > 2$) histopathology tasks where class discordance and epistemic vacuity can diverge.
3. **Retrospective Benchmark Bias:** Evaluation is confined to retrospective challenge cohorts (Camelyon17 and MIDOG 2021). Prospective, multi-scanner clinical trials are required before considering test-time adaptation for clinical diagnostic pathology.
4. **Computational Latency:** MC-Dropout requires 20 forward passes per patch ($598.2\,\text{s}$ vs. $168.7\,\text{s}$ for confidence gating on 320 patches). While acceptable for offline research, it presents an operational bottleneck for high-throughput whole-slide gigapixel scanners.

---

## Lessons Learned

1. **Negative Results Are High-Value Science:** Demonstrating that MC-Dropout variance is rank-equivalent to confidence in binary classification ($\rho = -0.9917$) eliminates wasteful engineering and protects the field from redundant complexity.
2. **Do Not Discard Collapsing Seeds:** Tent representation collapse on Seeds 7 and 99 revealed fundamental structural vulnerabilities that would have remained hidden under single-seed reporting.
3. **Biological Context Dictates Algorithmic Thresholds:** An algorithmic alarm calibrated on dense synthetic noise cannot be blindly deployed on sparse biological detection without accounting for tissue composition (acellular stroma).
4. **Permutation Invariance is Mandatory:** In clinical practice, patient arrival order is non-deterministic. Without episodic resets at slide boundaries, model predictions depend on patient sequencing.

---

## Future Work

- **Multi-Class Evidential Adaptation:** Extending uncertainty gating to multi-class tissue classification (e.g., NCT-CRC-HE-100K 9-class tissue typing) via Dirichlet Prior Networks to investigate true epistemic-aleatoric decoupling.
- **Cellularity-Aware Detection Safeguards:** Conditioning proposal silence alarms on Hematoxylin Optical Density ($H\text{-OD}$) via Ruifrok-Johnston color deconvolution to distinguish physiological acellular stroma from true detector collapse.
- **Whole-Slide Density Scaling:** Scaling test-time adaptation from patch-level sampling budgets (64--128 patches/WSI) to full gigapixel whole-slide scans ($20{,}000$--$50{,}000$ patches per slide).

---

## Project Structure

```text
UTTA-Med/
├── checkpoints/             # Verified source model checkpoints (SHA-256 audited)
├── configs/                 # YAML configuration files (camelyon17.yaml, v2/ configs)
├── data/                    # Dataset metadata and COCO annotation files (MIDOG.json)
├── docs/                    # Exhaustive audit reports, clinical protocols, and claims ledgers
│   ├── final/               # Finalized publication dossiers (SUBMISSION_READINESS.md)
│   └── v2/                  # Stage B through E2 execution specifications
├── figs/                    # Synchronized high-resolution publication figures (300 DPI)
├── notebooks/               # Standalone Kaggle runners and verification notebooks
│   └── official/            # Official qualified Kaggle workers
├── paper/                   # Manuscript drafts, IEEE LaTeX sources, and BibTeX files
│   ├── ieee/                # IEEE conference submission package (utta_med_h5_manuscript_draft.tex)
│   └── tables/              # Canonical publication tables in CSV and Markdown
├── reproducibility/         # Environment manifests, requirements lockfiles, and SHA-256 digests
├── results/                 # Authoritative machine-readable JSON and CSV experimental artifacts
│   ├── figures/             # Evaluation plots, reliability overlays, and AUROC curves
│   ├── generalization/      # MIDOG Stage H4 Phase A & Phase B detection benchmarks
│   ├── metrics/             # Headline frozen multi-seed metrics (full_s*.json)
│   ├── predictions/         # Serialized patch-level prediction arrays (s42_*.npz)
│   └── tables/              # Bit-exact CSV summary tables (table_main_results_final.csv)
├── scripts/                 # Execution, calibration, adaptation, and table verification scripts
│   └── v2/                  # Stage C2, D2, E1, E2 evaluation and audit scripts
├── src/                     # Core Python library
│   ├── evaluation/          # Metrics, coverage-risk, slide aggregation, and tent collapse
│   ├── models/              # ResNet-18 and ResNet-50 architectures
│   ├── tta/                 # Gating routers, Tent, EATA, and uncertainty scoring
│   └── v2/                  # SafeAdapt controller, boundary monitors, and telemetry
└── tests/                   # Pytest verification suite (137 unit and integration tests)
```

---

## References

1. **D. Wang et al.**, *"Tent: Fully Test-Time Adaptation by Entropy Minimization,"* in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
2. **S. Niu et al.**, *"Efficient Test-Time Model Adaptation without Forgetting,"* in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2022.
3. **P. W. Koh et al.**, *"WILDS: A Benchmark of in-the-Wild Distribution Shifts,"* in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021.
4. **P. Bandi et al.**, *"From Detection of Individual Metastases to Classification of Lymph Node Staging: The CAMELYON17 Challenge,"* *IEEE Trans. Med. Imaging*, vol. 38, no. 2, pp. 550–560, 2019.
5. **M. Aubreville et al.**, *"Mitosis Domain Generalization in Histopathology Images — The MIDOG Challenge,"* *Med. Image Anal.*, vol. 84, p. 102699, 2023.
6. **Y. Gal and Z. Ghahramani**, *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,"* in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2016.
7. **C. Guo et al.**, *"On Calibration of Modern Neural Networks,"* in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017.
8. **J. Liang et al.**, *"A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts,"* *arXiv:2303.15361*, 2023.

---

## License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for complete details.

---

## Acknowledgments

- **WILDS Benchmark Team:** For providing the standardized Camelyon17 multicenter histopathology dataset.
- **MIDOG Challenge Organizers:** For the multi-scanner mitotic figure detection dataset and evaluation frameworks.
- **Principal Author & Lead Researcher:** Avinash Reddy Cheerapareddy (`avinashreddych7@gmail.com`).
