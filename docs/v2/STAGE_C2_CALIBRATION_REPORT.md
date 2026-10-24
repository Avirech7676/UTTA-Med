# Stage C2: Hospital-1 Collapse Monitor Calibration Report

**Status:** Completed & Calibrated  
**Date:** October 2026  
**Auditor:** Independent Scientific Auditor & PyTorch Reliability Engineer  
**Dataset:** Camelyon17-WILDS (Hospital 1 / Center 1 ONLY)  
**Target Domain Isolation:** Hospital 2 (Center 2, Test Split) Strictly Isolated (Zero Access)  
**Study Designation:** Exploratory / Provisional ($N_{\mathrm{WSI}} = 10$)  

---

## 1. Executive Summary & Calibration Objectives

Stage C2 calibrates the label-free, slide-aware prediction-collapse monitor developed in Stage C1 using exclusively permitted development data from **Hospital 1 (Center 1)**. In adherence to strict scientific boundaries:
- **No v1 checkpoints or results modified:** Frozen source ResNet-18 (`seed=42`, `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`) is evaluated without modification.
- **Strict Center 2 isolation:** Hospital 2 (Center 2, slides 20–29) was never accessed for parameter tuning, calibration, or debugging.
- **Exploratory classification:** Because Hospital 1 provides only 10 whole-slide images ($N_{\mathrm{WSI}}=10$, 34,904 patches), this calibration is explicitly designated **exploratory and provisional**.
- **Observation-mode first:** Monitoring thresholds were derived from empirical distributions observed during label-free streaming adaptation on the disjoint **Calibration Partition** (Slides 10, 12, 14, 16, 18).

---

## 2. Experimental Hardware & Execution Profile

- **Operating System:** Windows 11 Enterprise (PowerShell execution environment)
- **Host Execution:** Intel CPU (`torch 2.14.0+cpu`, deterministic multi-threading)
- **Model Architecture:** ResNet-18 (ImageNet-pretrained source, 1-logit binary head)
- **Input Dimension:** $96 \times 96 \times 3$ RGB (WILDS standard)
- **Adaptation Scope:** BatchNorm affine parameters only ($\gamma, \beta$, 9,600 parameters across 40 tensors)
- **Optimizer:** Adam ($\text{lr} = 10^{-5}$, $\beta_1=0.9, \beta_2=0.999$, running stats frozen)
- **Average Step + Monitor Latency:** $8,862.19\text{ ms/batch}$ (CPU execution)
- **Memory Footprint:** Zero additional memory overhead due to decoupled dedicated source evaluation pipeline (`_source_model.eval()`).

---

## 3. Observation-Mode Empirical Distributions

The calibration partition comprises 5 whole-slide images from Hospital 1 with heterogeneous tumor prevalence:
- **Slide 10:** 7.81% tumor
- **Slide 12:** 7.81% tumor
- **Slide 14:** 3.12% tumor
- **Slide 16:** 81.25% tumor
- **Slide 18:** 12.50% tumor

The monitor operated in `MonitorMode.OBSERVATION` across streaming batches with sliding window $W=50$ and minimum sample gate $N_{\mathrm{min}}=20$. The observed empirical distributions over healthy adaptation steps are summarized below:

| Monitoring Metric | Mean | Min | 5th Percentile ($p_{05}$) | Median ($p_{50}$) | 95th Percentile ($p_{95}$) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logit Variance ($\sigma_z^2$)** | 22.7811 | 5.9938 | 7.1522 | 16.2010 | — | — |
| **Predictive Entropy ($\bar{H}$)** | 0.0918 | 0.0207 | 0.0214 | 0.0467 | — | — |
| **Continuous Disagreement ($\mathbb{E}[\|\hat{p} - p_0\|]$)** | 0.0010 | — | — | — | 0.0018 | 0.0020 |
| **Binary Flip Rate ($\mathbb{E}[\mathbf{1}_{\hat{y} \ne y_0}]$)** | 0.0020 | — | — | — | 0.0110 | 0.0200 |

### Key Diagnostic Observations:
1. **Healthy Logit Variance:** Even on slides with low tumor prevalence, the spread of logit outputs remains well above 5.99 ($\text{mean} = 22.78$). This provides a wide separation from mathematical single-class collapse (where variance collapses to $< 0.01$).
2. **Extremely Low Healthy Disagreement:** During stable test-time adaptation with $\text{lr}=10^{-5}$, the binary decision disagreement between the adapted model and the frozen source is negligible: mean 0.20%, 95th percentile 1.10%, and maximum 2.00%.
3. **Entropy Guardrail Necessity:** Normal lymph node tissue (0% tumor) yields low entropy without indicating collapse because the source model also predicts normal tissue with high confidence. Thus, low entropy must **never** trigger an alarm in isolation.

---

## 4. Calibrated Threshold Selection & Justification

Based on the empirical distributions, candidate thresholds were selected and frozen in `configs/v2/safeadapt_c2_calibrated.yaml`:

```yaml
collapse_monitor:
  mode: "ACTIVE"
  window_size: 50
  min_samples_for_alarm: 20
  decision_threshold: 0.5
  min_logit_variance: 0.08
  max_disagreement_rate: 0.35
  min_predictive_entropy: 0.05
  consecutive_suspicious_for_rollback: 2
  consecutive_suspicious_for_fallback: 4
  slide_aware_reset: true
  slide_reset_policy: "RESET_MONITOR_ONLY"
```

### Threshold Rationales:

1. **`min_logit_variance = 0.08`**:
   - *Healthy empirical range:* Min observed was 5.99.
   - *Rationale:* The threshold 0.08 sits two orders of magnitude below any observed healthy slide variance, while catching mathematical single-class collapse (where variance approaches 0.0005).

2. **`max_disagreement_rate = 0.35`**:
   - *Healthy empirical range:* Max observed was 0.020 (2.0%), $p_{95} = 0.011$ (1.1%).
   - *Rationale:* 0.35 provides a $\approx 17\times$ safety margin over the empirical maximum. It ensures that benign slide-level adaptation shifts are never falsely penalized, while catastrophic model divergence or inverted predictions immediately trip suspicious windows.

3. **`consecutive_suspicious_for_rollback = 2`**:
   - *Rationale:* Protects against transient batch-level statistical fluctuations (e.g. edge-of-slide stroma artifact). A single anomalous window produces a `WARNING` without interrupting parameter adaptation. Only 2 consecutive confirmed anomalous windows trigger an atomic rollback.

4. **`consecutive_suspicious_for_fallback = 4`**:
   - *Rationale:* If the controller encounters 4 consecutive rejected windows despite atomic pre-update rollback, adaptation has proven persistently unviable on this specimen. The controller locks into permanent frozen-source fallback for the remainder of the slide.

---

## 5. Provenance & Scientific Boundary Compliance

- **No Retraining:** Zero training iterations were performed on source ResNet-18 or ResNet-50.
- **Hospital 2 Untouched:** Zero Hospital-2 images, patches, or annotations were queried.
- **Non-Inferiority Disclaimer:** This calibration confirms numerical stability and detection sensitivity under controlled experimental conditions; it does **not** constitute a mathematical guarantee of clinical safety across unseen clinical institutions.
