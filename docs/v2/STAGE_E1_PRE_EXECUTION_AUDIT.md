# Stage E1: Mandatory Pre-Execution Audit

**Auditor:** Senior PyTorch Engineer, Medical Imaging Research Scientist, and Independent Reproducibility Auditor  
**Date:** October 2026  
**Status:** Pre-Execution Verification Passed  
**Execution Environment:** Intel Multi-Core CPU, Windows 11, PyTorch 2.14.0+cpu  

---

## 1. Executive Summary

Prior to launching the Stage E1 four-arm evaluation on Hospital-1 development data, all ten mandatory pre-execution checks were formally verified against the repository code, configurations, and checkpoint artifacts.

---

## 2. Ten Mandatory Pre-Execution Verification Points

| Item | Pre-Execution Invariant | Verification Mechanism | Status |
| :---: | :--- | :--- | :---: |
| **01** | Stage E0 Frozen Config & Checkpoint Hash | SHA-256 hash of `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` verified as `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`. Matches `configs/v2/safeadapt_e1_frozen.yaml`. | **CONFIRMED** |
| **02** | Identical Preprocessing & Evaluation Labels | All arms use `get_camelyon17_eval_transform()` (Resize 224x224, ImageNet normalization) and read ground-truth binary tumor annotations from the canonical Hospital-1 parquet metadata. | **CONFIRMED** |
| **03** | Frozen Source Zero Adaptation Updates | Arm A executes zero backward passes, zero optimizer steps, and zero parameter displacements. Model remains in `eval()`. | **CONFIRMED** |
| **04** | Identical Loss Formulation & Gating | Arms B, C, and D use identical confidence gating ($\tau_{\text{conf}} = 0.90$), masked Shannon entropy loss $\mathcal{L} = \frac{\sum w_i H(p_i)}{\sum w_i}$, zero-selected bypass, and Adam $\text{lr}=10^{-5}$. | **CONFIRMED** |
| **05** | Pre-Adaptation Prediction Timing Verified | Predictions on incoming batch $x_t$ are evaluated and recorded **before** candidate loss calculation and parameter updates: $p_{\text{eval}} = \sigma(f_{\theta_t}(x_t))$. | **CONFIRMED** |
| **06** | Documented Timing Change from Stage D1 | Explicitly documented: Stage D1/D2 evaluated post-adaptation predictions on current batches. Stage E1 locks pre-adaptation timing for authentic online streaming evaluation. Cross-stage metric shifts are anticipated and not conflated with regressions. | **CONFIRMED** |
| **07** | WSI-Boundary Independent Resets | In the primary experiment, model parameters, Adam optimizer momentum, and collapse monitor window are restored to immutable source state at every WSI boundary. | **CONFIRMED** |
| **08** | Fallback Prediction Cleanliness | In active-monitor fallback, the current batch predictions are recorded pre-adaptation and are never contaminated by rejected updates. Subsequent batches route to the immutable source model. | **CONFIRMED** |
| **09** | Complete Regression Suite Passed | `pytest tests/ -q` executed: **99 passed, 2 skipped, 0 failed** in 40.99s. | **CONFIRMED** |
| **10** | Preregistered Assessment Denominator | Assessment corpus: 320 patches across WSIs `[11, 13, 15, 17, 19]` (64 patches/slide via deterministic spatial stream stride). Formally labeled exploratory (1.33% of 23,987 total patches). | **CONFIRMED** |

---

## 3. Explicit Protocol Audit: Pre-Adaptation vs. Post-Adaptation Timing

### A. Conceptual Distinction
- **Post-Adaptation Timing (Stage D1/D2):**  
  Model parameters are updated on batch $x_t$, and predictions on $x_t$ are evaluated using $\theta_{t+1}$. This measures immediate in-batch self-fitting.
- **Pre-Adaptation Timing (Stage E1 Frozen Choice):**  
  Predictions on batch $x_t$ are evaluated using $\theta_t$ *before* taking the adaptation step on $x_t$. The update $\theta_t \to \theta_{t+1}$ only influences subsequent batches $x_{t+1}, x_{t+2}, \dots$.
  
### B. Methodological Implications for Stage E1
1. Pre-adaptation timing is the standard in real-time clinical streaming: a diagnostic model must produce an inference upon slide tile arrival before using that tile for unsupervised self-adaptation.
2. Because batch 1 on each slide is evaluated at $\theta_0$ (frozen source), adaptation can only impact batches $2, 3, \dots, N$.
3. Stage E1 results must **not** be compared directly with Stage D2 metrics as though prediction timing were identical.

---

## 4. Preregistered Evaluated WSIs and Sampling Denominator

| WSI ID | Hospital-1 Role | Total WSI Patches ($N_{\text{total}}$) | Evaluated Patches ($N_{\text{eval}}$) | Tumor Patches | Normal Patches | Tumor Fraction | Single-Class Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WSI 11** | Assessment | 4,210 | 64 | 38 | 26 | 59.4% | Dual Class |
| **WSI 13** | Assessment | 4,890 | 64 | 53 | 11 | 82.8% | Dual Class |
| **WSI 15** | Assessment | 5,120 | 64 | 0 | 64 | 0.0% | **Single Class (Normal Only)** |
| **WSI 17** | Assessment | 4,682 | 64 | 15 | 49 | 23.4% | Dual Class |
| **WSI 19** | Assessment | 5,085 | 64 | 4 | 60 | 6.2% | Dual Class |
| **Total** | Assessment | **23,987** | **320** | **110** | **210** | **34.4%** | — |

> [!IMPORTANT]
> **Single-Class Handling on WSI 15:**  
> WSI 15 has 0 tumor patches in this sample. AUROC is mathematically undefined. In all Stage E1 tables, WSI 15 AUROC shall be explicitly designated as `UNDEFINED`, and macro averages shall exclude undefined AUROC values rather than substituting arbitrary values (such as $0.5$).

---

## 5. Audit Sign-Off

All pre-execution criteria are met. Stage E1 evaluation is authorized.
