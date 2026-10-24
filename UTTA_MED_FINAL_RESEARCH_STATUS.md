# UTTA-MED PROJECT 2: FINAL RESEARCH STATUS REPORT

**Official Research Title:**  
*Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*

**Branch:** `research-project-2` | **Audit Date:** October 2026  
**Auditor / Lead Engineer:** Senior Computational Pathology Researcher, Statistical Auditor & Reproducibility Specialist  
**Deliverable Type:** Final Executive Research Status Report

---

## 1. RESTORATION STATUS & RESEARCH ALIGNMENT

The UTTA-Med Project 2 codebase and documentation have been **fully brought back to the original approved research direction** defined by the authoritative research manuscript ([`UTTA_MED_FINAL_COMPACT_PAPER.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.tex)):

1. **Research Identity Preserved:** The official research title, problem formulation, and two primary research tracks (Track A: Camelyon17-WILDS multicenter classification; Track B: MIDOG 2021 multi-scanner detection) are permanently locked.
2. **Backbone Architectures Verified:**
   - Classification: Single-logit ResNet-18 ($11.45\text{M}$ parameters) with BCE loss in [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py).
   - Detection: Faster R-CNN with ResNet-50 FPN backbone on MIDOG 2021 Arms A–E.
3. **No Replacement Architectures:** DenseNet-121, ResNet-50 for classification, and EMA Teacher-Student learning remain safely quarantined on branch `archive/unrequested-research-extensions` and external archive `C:/Users/avina/Documents/UTTA_MED_RESEARCH_ARCHIVE/`.
4. **Authentic Adaptation Active:** All evaluation shortcuts (`np.where`) are permanently excised. The active canonical validation pipeline executes real PyTorch backpropagation and Adam optimizer steps on 40 BatchNorm affine parameters ($\Delta\theta_{L_2} = 0.007026$).
5. **Safeguards & Resets Operational:** Verified exact parameter restoration ($\Delta\theta = 0.000000$) across 20 slide transitions and 100% rollback success across 5 fault injection suites.

---

## 2. REPRODUCIBILITY STATUS

All primary results are reproduced directly from certified source checkpoints and raw unrounded evaluation arrays:

1. **Source Checkpoints:** All 5 Camelyon17 ResNet-18 checkpoints (Seeds $\{42, 123, 2024, 7, 99\}$) and the MIDOG Faster R-CNN checkpoint match canonical SHA-256 hashes recorded in [`PRESERVED_CORE_FILES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/PRESERVED_CORE_FILES.csv).
2. **Prediction Arrays:** All 35 target prediction files in `results/data/wsi_bootstrap/` match cryptographic SHA-256 hashes recorded in [`TARGET_PREDICTION_PROVENANCE.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/TARGET_PREDICTION_PROVENANCE.md).
3. **Automated Regression Suite:** Pytest execution confirms **77 passed, 5 skipped, 0 failed** in 26.13 seconds with zero test regressions.

---

## 3. INVESTIGATION OF THE ORIGINAL PERFORMANCE OBJECTIVE

### 3.1 The Comparison & Failure Analysis
The primary performance question in the original research was:  
*Why does Monte Carlo Dropout uncertainty gating (Original UTTA-Med, $F_1 = 0.8296 \pm 0.0325$) fail to outperform simple scalar Confidence Gating ($A_1, F_1 = 0.8310 \pm 0.0321$, $p = 0.006202$)?*

Our forensic investigation revealed two underlying mechanisms:
1. **Monotonic Alignment in Binary Classification:** For a single-logit model with sigmoid activation $p(x) = \sigma(z(x))$, epistemic head-dropout variance and aleatoric confidence $\max(p, 1-p)$ are nearly monotonically aligned near the decision threshold $p \approx 0.5$. Hard variance filtering does not identify a distinct set of informative patches compared to thresholding on prediction confidence.
2. **Tumor Patch Scarcity & Asymmetric Exclusion:** In whole-slide images with extreme tumor scarcity (e.g., Slide 21 with $0.27\%$ tumor prevalence), symmetric uncertainty gating rejects borderline tumor patches, starving the model of positive adaptation signals.

### 3.2 Validation-First Controlled Improvement & Decision Gate
To test whether this limitation could be overcome while strictly preserving the original ResNet-18 and MC-Dropout framework:
- We evaluated **Class-Aware Dual Gating (CAD-Gate)** on held-out Center 1 validation data ($N=600$ patches across 10 WSIs) in [`scripts/run_final_controlled_improvement.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/run_final_controlled_improvement.py).
- On Center 1 validation, CAD-Gate achieved $F_1 = 0.8625$, exactly matching Confidence Gating ($0.8625$) and Original UTTA-Med ($0.8625$), producing the identical 3 hard prediction flips.
- **Decision Gate Outcome:** Because CAD-Gate demonstrated zero diagnostic accuracy gain over Confidence Gating on validation data, it was **not** promoted to the 5-seed Center 2 target benchmark.
- **Computational Optimization:** Feature-cached MC-Dropout ($A_3$) was evaluated across all 5 seeds on Center 2. While it reduced latency from $136.29$ ms to $7.82$ ms ($17.4\times$ speedup), its diagnostic $F_1$ remained at $0.8303 \pm 0.0322$, failing to surpass Confidence Gating ($0.8310$).

### 3.3 Preservation of the Negative Result
Under Occam's razor, the scientific finding stands:
**Monte Carlo Dropout predictive variance imposes an unnecessary $17.4\times$ to $20.2\times$ computational overhead in single-logit binary classification while providing zero diagnostic accuracy advantage over scalar confidence gating.** We report this negative result with full scientific integrity rather than inventing ad-hoc replacement architectures to force an artificial positive gain.

---

## 4. CANONICAL BENCHMARK PERFORMANCE

### 4.1 Track A: Camelyon17-WILDS Classification (5 Seeds, $N=85{,}054$ Patches)

| Arm | Method | Binary $F_1$ (Mean ± SD) | Target AUROC (Mean ± SD) | Coverage (%) | Harm / Cohort ($H/N$) | Harm / Accepted ($H/k$) | Net Flips ($B - H$) | Batch Latency | Decision Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$A_0$** | **Frozen Source ResNet-18** | $0.8069 \pm 0.0247$ | $0.9297 \pm 0.0092$ | $0.0\%$ | N/A ($0 / N$) | N/A | $0$ (Ref) | **6.74 ms** | Source Baseline |
| **Tent** | Unconstrained Entropy Min | $0.5057 \pm 0.4111$ | $0.6981 \pm 0.3237$ | $100.0\%$ | $15.48\%$ ($13{,}168$) | $15.48\%$ | $-10{,}790.6$ | 7.70 ms | Collapsed (2/5 seeds) |
| **EATA** | Point Entropy Filtered | $0.8096 \pm 0.0316$ | $0.9215 \pm 0.0263$ | $88.5\%$ | $1.42\%$ ($1{,}211$) | $1.61\%$ | $+458.4$ | 7.75 ms | Inferior to $A_1$ |
| **$A_1$** | **Confidence Gating ($\tau=0.85$)** | $\mathbf{0.8310 \pm 0.0321}$ | $\mathbf{0.9355 \pm 0.0218}$ | $\mathbf{92.88\%}$* | $\mathbf{0.585\%}$ ($497.4$) | $\mathbf{0.630\%}$ | $\mathbf{+1{,}741.8}$ | **7.68 ms** | **Preferred Method** |
| **$A_2$** | **Original UTTA-Med (MC-Dropout)** | $0.8296 \pm 0.0325$ | $0.9351 \pm 0.0218$ | $92.87\%$* | $0.612\%$ ($520.2$) | $0.659\%$ | $+1{,}659.8$ | 136.29 ms | Inefficient ($p=0.0062$) |
| **$A_3$** | **Improved UTTA-Med (Feature-Cached)** | $0.8303 \pm 0.0322$ | $0.9351 \pm 0.0217$ | $92.87\%$ | $0.598\%$ ($508.6$) | $0.644\%$ | $+1{,}702.8$ | 7.82 ms | Redundant vs $A_1$ |

*\*Under matched-budget operating constraint ($58.1\%$ coverage), $A_1$ yields 595 harmful errors ($0.700\%$ of cohort, $1.204\%$ of accepted patches).*

### 4.2 Track B: MIDOG 2021 Multi-Scanner Detection (3 Seeds, 50 Cases, 1,100 Patches)

| Arm | Method Description | Precision | Recall | $F_1$ Score | TP | FP | FN | Rollbacks / Fallbacks | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Arm A** | **Frozen Source Baseline** | 0.4916 | 0.4718 | **0.4774** | 55.7 | 59.3 | 62.3 | 0 / 0 | Source Baseline |
| **Arm B** | **Conventional TTA (Tent)** | 0.5597 | 0.4096 | 0.4702 | 48.3 | 39.3 | 69.7 | 0 / 0 | Divergent (177.3 alarms) |
| **Arm C** | **Bounded TTA ($\|\Delta\theta\|_\infty \le 0.05$)** | 0.5597 | 0.4096 | 0.4702 | 48.3 | 39.3 | 69.7 | 0 / 0 | Parameter Bounded |
| **Arm D** | **Monitored TTA (Atomic Rollback)** | 0.5612 | 0.4096 | 0.4709 | 48.3 | 39.0 | 69.7 | 0.7 / 0 | Rollback Active |
| **Arm E** | **SafeAdapt (Dual Recovery)** | 0.4916 | 0.4718 | **0.4774** | 55.7 | 59.3 | 62.3 | 1.0 / 1.0 | Latched Source Fallback |

---

## 5. REMAINING SCIENTIFIC LIMITATIONS

1. **Target Evaluation Scope:** Camelyon17-WILDS target testing evaluates 10 WSIs from 9 patients from a single medical center (Center 2). Testing across 5 training seeds evaluates optimization stability across weight initializations, but cannot establish multi-institution patient population generalization.
2. **Absence of Ground-Truth Lesion Diameters:** WILDS patch annotations provide central $32 \times 32$ binary tumor labels, not continuous lesion diameters or institutional TNM stages.
3. **Detection Adaptation Behavior:** SafeAdapt preserves baseline performance ($\Delta F_1 = 0.0000$) through early latching fallback to source weights, containing failure rather than achieving positive adaptation.
4. **Clinical Deployment Status:** This system is a research prototype. It is **not** approved for clinical diagnostic use.

---

## 6. CALIBRATED SCIENTIFIC READINESS VERDICT

We separate readiness into six distinct scientific dimensions:
1. **Implementation Correctness:** **100% VERIFIED.** Authentic PyTorch gradients, zero shortcut leakage, verified slide boundary resets, and active fail-closed rollbacks.
2. **Benchmark Results:** **REPRODUCIBLE.** Authenticated against raw prediction arrays across 5 random seeds.
3. **Statistical Conclusions:** **ROBUST WITHIN SEEDS.** Paired comparisons confirm consistent seed-level improvement for Confidence Gating ($p = 0.006202$).
4. **Remaining Scientific Limitations:** Documented transparently without concealment.
5. **Manuscript Submission Readiness:** **SUBMISSION READY.** Transparent methodology, complete negative results preservation, and full code reproducibility suitable for peer-reviewed journal submission to IEEE TMI, Medical Image Analysis, or MICCAI.
6. **Clinical Deployment Readiness:** **NOT CLINICALLY VALIDATED.** Clinical deployment requires prospective multi-institution clinical trials and regulatory authorization.
