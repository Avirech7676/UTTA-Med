# Stage D1: Audit of Original v1 Gating Semantics

**Auditor:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Reproducibility Auditor  
**Date:** October 2026  
**Status:** Complete  
**Reference Codebase:** `src/tta/gates.py`, `src/tta/uncertainty_gated.py`, `src/models/resnet18.py`, `configs/tau_by_seed.yaml`  

---

## 1. Executive Summary

This forensic audit analyzes the test-time adaptation gating semantics of the frozen UTTA-Med v1 codebase to establish the definitive reference specifications for Stage D1 faithful implementation. 

Eight fundamental questions regarding v1 gating semantics are resolved through direct source-code inspection.

---

## 2. Forensic Audit of Eight Gating Dimensions

### 2.1 Confidence Computation
- **Source:** `src/tta/gates.py`, lines 15–21 (`score_confidence`).
- **Mathematical Definition:**
  $$p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$
  $$\text{conf}_i = \max(p_i, 1 - p_i) \in [0.5, 1.0]$$
- **Implementation:** Evaluated deterministically in `model.eval()` mode with zero dropout passes.
- **Decision Rule:** Patch $i$ is selected if $\text{conf}_i > \tau_{\mathrm{conf}}$.

### 2.2 MC-Dropout Uncertainty Computation
- **Source:** `src/tta/gates.py`, lines 36–53 (`score_mc_dropout`) and `src/models/resnet18.py`, lines 82–90 (`mc_predict`).
- **Mathematical Definition:**
  For $k \in \{1, \dots, K\}$ stochastic passes with dropout enabled:
  $$p_{i,k} = \sigma(z_{i,k})$$
  $$\bar{p}_i = \frac{1}{K} \sum_{k=1}^K p_{i,k}$$
  $$U_i = \text{Var}(p_{i,k}) = \frac{1}{K} \sum_{k=1}^K (p_{i,k} - \bar{p}_i)^2 \quad (\text{unbiased}=\text{False})$$
- **Decision Rule:** Patch $i$ is selected if $U_i < \tau_{\mathrm{unc}}$.

### 2.3 Selection Granularity: Patch-Level vs. Batch-Level
- **Source:** `src/tta/uncertainty_gated.py`, lines 57–76 (`_keep_mask`).
- **Granularity:** Strictly **PATCH-LEVEL**.
- **Audit Finding:** The selection mask `keep` has shape `(B,)` where each element represents an individual patch $i$. Gating is **never** averaged across the batch. Any implementation replacing patch-level filtering with batch-mean thresholding violates v1 scientific equivalence.

### 2.4 Uncertainty Threshold Calibration
- **Source:** `configs/tau_by_seed.yaml`, lines 1–18.
- **Calibration Protocol:** $\tau_{\mathrm{unc}}$ is calibrated as the empirical **70th percentile of Hospital-1 validation predictive variance $U$**, pre-computed per seed on permitted development data:
  - Seed 42: $\tau = 8.875383 \times 10^{-6}$
  - Seed 123: $\tau = 7.098636 \times 10^{-6}$
  - Seed 2024: $\tau = 1.293730 \times 10^{-5}$
  - Seed 7: $\tau = 1.198491 \times 10^{-5}$
  - Seed 99: $\tau = 1.829144 \times 10^{-5}$
- **Negative Governance Notice:** $\tau_{\mathrm{unc}}$ is on the variance scale ($\sim 10^{-5}$), never coverage scale (e.g. 0.05). Thresholds are strictly frozen and never retuned on Hospital 2.

### 2.5 Masked Adaptation Loss Contribution
- **Source:** `src/tta/uncertainty_gated.py`, lines 84–94 (`adapt_batch`).
- **Mathematical Definition:**
  For binary Shannon entropy $H(p_i) = -(p_i \log p_i + (1-p_i) \log (1-p_i))$ and selection mask $w_i \in \{0, 1\}$:
  $$\mathcal{L} = \frac{\sum_{i=1}^B w_i \cdot H(p_i)}{\sum_{i=1}^B w_i}$$
- **Zero-Patch Behavior:** If $\sum_{i=1}^B w_i = 0$ (no patches meet the gating criteria), **the entire optimization step is skipped** (0 backward passes, 0 optimizer updates). The batch is counted as skipped in telemetry (`n_skipped_batches`).

### 2.6 Number of Stochastic Passes
- **Source:** `configs/tau_by_seed.yaml`, line 6 (`n_mc: 20`) and `src/tta/uncertainty_gated.py`, line 30 (`n_passes: int = 20`).
- **Specification:** Exactly $K = 20$ stochastic forward passes per patch batch.

### 2.7 Inference Timing
- **Source:** `src/tta/uncertainty_gated.py`, lines 95–97.
- **Specification:** Model predictions returned to the caller are computed **after adaptation** on the current batch:
  $$\hat{y}_{\mathrm{eval}} = \sigma(\text{model}_{\mathrm{adapted}}(x)).$$

### 2.8 Dropout and BatchNorm Configuration
- **Source:** `src/tta/bn.py`, lines 104–120 (`configure_tta_mode`) and `src/models/resnet18.py`, lines 70–75 (`enable_dropout`).
- **BatchNorm Mode:** All BatchNorm layers have `running_mean` and `running_var` frozen (`bn.eval()`) during both MC passes and adaptation. Only affine parameters ($\gamma, \beta$) have `requires_grad=True`.
- **Dropout Mode:**
  - During MC inference: `model.eval()`, followed by `dropout.train()` on Dropout layers only.
  - During candidate adaptation step: `model.eval()` on Dropout layers (deterministic forward pass for loss computation).
  - During final evaluation: `model.eval()` on all layers.

---

## 3. Reference Policy Summary Matrix

| Dimension | Policy A: Frozen Source | Policy B: Confidence-Only | Policy C: MC-Only (v1 Faithful) | Policy D: Efficient Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| **Gating Metric** | None | $\max(p, 1-p)$ | Predictive Variance $U$ | Two-stage: Confidence $\to$ $U$ |
| **Selection Scope** | Zero patches | Patch-level ($c_i > \tau_{\mathrm{conf}}$) | Patch-level ($U_i < \tau_{\mathrm{unc}}$) | Patch-level ($c_i \ge \theta_{\mathrm{hi}}$ or ambiguous $U_i < \tau_{\mathrm{unc}}$) |
| **Stochastic Passes** | 0 passes | 0 passes | 20 passes (100% of patches) | 20 passes (ambiguous subset only) |
| **Loss Masking** | N/A | Masked Entropy | Masked Entropy | Masked Entropy |
| **Zero-Selection Action** | Skip update | Skip update | Skip update | Skip update |
| **Post-Step Inference** | Frozen source | Adapted eval | Adapted eval | Adapted eval |
| **BN Running Stats** | Frozen | Frozen | Frozen | Frozen |
