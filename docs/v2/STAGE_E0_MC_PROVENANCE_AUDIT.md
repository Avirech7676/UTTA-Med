# Stage E0: MC-Dropout Threshold Provenance and Configuration Audit

**Auditor:** Independent Scientific Auditor, Senior PyTorch Engineer, and Medical Imaging Reproducibility Researcher  
**Date:** October 2026  
**Status:** Provenance Verified & Confirmed  
**Target:** MC-Dropout Predictive Variance Gating Threshold ($\tau$)  

---

## 1. Executive Summary

This audit traces the complete lineage, mathematical definition, calibration population, and software configuration of the MC-only threshold ($\tau_{\text{var}} = 8.87538 \times 10^{-6} \approx 8.88 \times 10^{-6}$) utilized in Stage D1 and Stage D2.

### Forensic Verdict
The threshold $\tau_{\text{seed42}} = 8.875383173290174 \times 10^{-6}$ is **fully authentic, traceable, and correctly matched** to:
1. Source checkpoint `camelyon17_resnet18_source_s42_BEST.pt` (SHA-256: `f0b85ec1a897...`, Seed 42).
2. The classifier dropout layer with $p=0.5$.
3. Exactly $K=20$ stochastic forward passes.
4. The population variance estimator $\text{Var}(p_k)$ with `unbiased=False`.
5. The 70th percentile of Hospital-1 validation predictive variance.

---

## 2. Parameter Lineage and Provenance Matrix

| Provenance Attribute | Target Protocol Requirement | Repository Evidence | Verification Status |
| :--- | :--- | :--- | :---: |
| **Model Checkpoint** | Frozen ResNet-18 Source Model | `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | **CONFIRMED** |
| **Random Seed** | Seed 42 (Source and Initializer) | `configs/tau_by_seed.yaml` (key `42`) | **CONFIRMED** |
| **Dropout Architecture** | Linear(512, 512) $\to$ ReLU $\to$ Dropout(p=0.5) $\to$ Linear(512, 1) | `Net.classifier[2]` in `scripts/v2/evaluate_stage_d2.py` | **CONFIRMED** |
| **Stochastic Passes ($K$)** | Exactly 20 stochastic passes | `configs/tau_by_seed.yaml` (`n_mc: 20`) | **CONFIRMED** |
| **Variance Estimator** | $\frac{1}{K}\sum_{k=1}^K (p_k - \bar{p})^2$ (Population Var) | `torch.var(P, dim=0, unbiased=False)` | **CONFIRMED** |
| **Calibration Rule** | 70th percentile of predictive variance ($U$) | `configs/tau_by_seed.yaml` (`rule: val_70th_percentile_of_U`) | **CONFIRMED** |
| **Calibration Set** | Hospital 1 (Center 1) validation patches | 34,904 patches across 10 WSIs | **CONFIRMED** |
| **Threshold Value** | $\tau = 8.875383173290174 \times 10^{-6}$ | `configs/tau_by_seed.yaml` line 10 | **CONFIRMED** |

---

## 3. Mathematical and Algorithmic Audit

### A. Variance Estimator Specification
Given an input patch $x_i$, $K=20$ stochastic forward passes produce sigmoid probabilities $\{p_{i, 1}, p_{i, 2}, \dots, p_{i, K}\}$. The MC-dropout predictive variance is computed as:
$$U(x_i) = \frac{1}{K} \sum_{k=1}^{K} \left( p_{i, k} - \bar{p}_i \right)^2, \quad \text{where } \bar{p}_i = \frac{1}{K} \sum_{k=1}^{K} p_{i, k}$$

> [!IMPORTANT]
> **Estimator Bias Semantics (`unbiased=False`):**  
> PyTorch's default `torch.var()` sets `unbiased=True`, dividing the sum of squared differences by $K - 1 = 19$ (Bessel's correction). In contrast, the v1 UTTA-Med specification uses population variance (`unbiased=False`, dividing by $K = 20$).  
> Dividing by 19 instead of 20 would inflate variance estimates by a factor of $1.0526\times$, incorrectly altering threshold crossings. In Stage D1 and Stage D2, `unbiased=False` was strictly enforced.

### B. BatchNorm Running Statistics Isolation
During stochastic forward sampling:
- Classifier dropout is explicitly switched to training mode: `module.train()`.
- All `BatchNorm2d` layers remain strictly frozen in evaluation mode: `module.eval()`.
- `requires_grad` is disabled on all parameters during sampling (`torch.no_grad()`).
- Verification in Stage D1 unit test `test_4_frozen_bn_statistics_during_mc_passes` confirmed that `running_mean` and `running_var` are bit-exact invariant across all 20 passes.

---

## 4. Empirical Consistency Check on Development WSIs

When $\tau = 8.87538 \times 10^{-6}$ was applied to the 320 calibration patches (WSIs `[10, 12, 14, 16, 18]`):
- **Selected Patches:** 231 out of 320 patches.
- **Observed Selection Fraction:** **$72.19\%$** ($\approx 72.2\%$).
- **Expected Theoretical Selection Fraction:** $70.00\%$.
- **Residual Difference:** $+2.19\%$ (7 patches out of 320).

This close correspondence ($72.19\%$ vs. $70.0\%$) verifies that the calibration on the 320-patch stratified sample behaves consistently with the nominal 70th percentile threshold derived from the full 34,904 Hospital-1 validation patches.

---

## 5. Potential Mismatches and Guardrails Documented

1. **Informal Documentation Discrepancy:**
   Historical narrative documents occasionally referred to $\tau = 0.05$ or $\tau = 0.70$. These refer to the *coverage quantile* ($70\%$) or *rejection rate* ($5\%-30\%$), not the variance magnitude. The actual variance threshold must always remain on the order of $\approx 8.88 \times 10^{-6}$.
2. **Cross-Seed Invariance:**
   $\tau$ is strictly seed-specific. For Seed 42, $\tau = 8.88 \times 10^{-6}$. For Seed 123, $\tau = 7.10 \times 10^{-6}$. For Seed 2024, $\tau = 1.29 \times 10^{-5}$. A single global threshold must never be applied across seeds.
3. **No Retuning on Hospital 2:**
   The threshold $\tau = 8.88 \times 10^{-6}$ was derived exclusively from Hospital 1 (Center 1) development data. It must never be adjusted or retuned using Hospital-2 (Center 2) test data.
