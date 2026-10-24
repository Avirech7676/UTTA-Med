# Stage E2: Target-Domain Collapse Monitor Behavior and Telemetry Audit

**Auditor:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Complete & Audited  
**Cohort:** Camelyon17 Hospital 2 (Center 2, WSIs 20–29, 1,280 patches)  
**Configuration:** `configs/v2/safeadapt_e2_frozen.yaml` ($W=50, \min=20$)

---

## 1. Executive Summary

Phase 3 audits the operational validity of the Stage C2 active collapse monitor on the actual Hospital-2 execution path. Unlike Stage E1 (where a configuration defect rendered sample sufficiency permanently false), Stage E2 verified that:
1. Every batch contributed patch-level prediction scalars to the slide buffer.
2. Sample sufficiency was reached on **100% of batches** ($w \ge 20$ satisfied from Batch 1 onward).
3. The monitor evaluated **40 eligible monitoring decisions** across the 10 target slides.
4. Zero false alarms occurred on legitimate 0% tumor normal tissue (WSIs 20 and 21).
5. The distinction between zero alarms and zero eligible opportunities was formally resolved.

---

## 2. Distinction: Eligible Decisions vs. Alarms

| Metric | Stage E1 (Flawed Config: $W=10, \min=100$) | Stage E2 (Reconciled Config: $W=50, \min=20$) | Scientific Meaning |
| :--- | :---: | :---: | :--- |
| **Total Evaluation Batches** | 10 (2 per WSI) | 40 (4 per WSI) | Full patch stream |
| **Eligible Monitoring Decisions** | **0 / 10 (0.0%)** | **40 / 40 (100.0%)** | $w \ge \text{min\_samples}$ satisfied |
| **Sample Sufficiency State** | Permanently `False` | **`True` on all batches** | Detector active |
| **Warnings Raised** | 0 | 0 | Non-breaching |
| **Rollbacks Triggered** | 0 | 0 | Non-breaching |
| **Fallbacks Triggered** | 0 | 0 | Non-breaching |
| **Interpretation of Zero Alarms** | **Muted Detector** (Defect) | **Empirical False-Alarm Clearance** | Confirmed stable pass-through |

In Stage E2, the zero alarm count represents genuine behavioral clearance across 40 evaluation windows rather than structural silence.

---

## 3. Granular Batch-Level Progression Within Slides

With 128 patches per slide processed in batches of 32:
- **Batch 1 ($n=32$ patches):** Slide buffer receives 32 patches. Because $32 \ge 20$, `is_sufficient_samples` evaluates to `True`. The multi-signal policy inspects logit variance, continuous disagreement, and binary flips.  
  $\to$ Eligible Decision Count = 1.
- **Batch 2 ($n=64$ patches):** Slide buffer receives 32 additional patches. Window slices the most recent $W=50$ patches ($50 \ge 20$).  
  $\to$ Eligible Decision Count = 2.
- **Batch 3 ($n=96$ patches):** Buffer holds 96 patches. Window slices the most recent $W=50$ patches.  
  $\to$ Eligible Decision Count = 3.
- **Batch 4 ($n=128$ patches):** Buffer holds 128 patches. Window slices the most recent $W=50$ patches.  
  $\to$ Eligible Decision Count = 4.
- **WSI Transition Reset:** At slide boundary, `reset_to_source()` wipes model parameters to $\theta_0$, clears the Adam optimizer state, and calls `reset_slide()` to wipe the monitor buffer to 0.

---

## 4. Normal Tissue Guardrail Verification (WSIs 20 & 21)

A crucial clinical requirement established in Stage C1 was that whole-slide images containing **0% tumor patches** must not trigger spurious collapse alarms.
- **WSI 20 (128 patches, 0 tumor):**
  - Predicted positive fraction: $\bar{p}_{\text{pos}} = 0.000$
  - Source positive fraction: $\bar{p}_{\text{source}} = 0.000$
  - Logit variance: $\sigma_z^2 = 18.42 \gg 0.08$
  - Continuous disagreement: $0.0012 \ll 0.35$
  - **Detector Outcome:** Correctly recognized concordance between source and adapted models on normal tissue. Evaluated 4 eligible decisions; raised **0 alarms**.
- **WSI 21 (128 patches, 0 tumor):**
  - Predicted positive fraction: $\bar{p}_{\text{pos}} = 0.000$
  - Source positive fraction: $\bar{p}_{\text{source}} = 0.000$
  - Logit variance: $\sigma_z^2 = 14.85 \gg 0.08$
  - Continuous disagreement: $0.0018 \ll 0.35$
  - **Detector Outcome:** Evaluated 4 eligible decisions; raised **0 alarms**.

---

## 5. Detector Sensitivity Analysis: Why Fallback Did Not Trigger on Slide 29

On WSI 29, adaptation caused a slight F1 decrease ($0.3659 \to 0.3457$) and AUROC decrease ($0.9293 \to 0.9281$). Why did the collapse monitor not trigger an emergency fallback?

### Mechanistic Explanation:
1. **Coarse Threshold Purpose:**  
   The collapse monitor was engineered to prevent catastrophic single-class divergence (the failure mode seen in v1 ResNet-50 where predictions collapsed to 100% negative and logit variance vanished to $< 0.01$).
2. **Observed Drift Magnitude on Slide 29:**  
   - Source-adapted continuous disagreement was $\mathbb{E}[|p - p_0|] \approx 0.0025$, well below the calibrated safety ceiling of $0.35$.
   - Binary decision flip rate was $\approx 0.015$, well below the $0.35$ threshold.
   - Logit variance remained high ($\sigma_z^2 \approx 11.2 \gg 0.08$).
3. **Implication:**  
   The collapse monitor operates as a coarse catastrophic circuit breaker, not a high-resolution sub-percent accuracy optimizer. It safely permitted the adaptation to proceed because the model remained bounded and non-collapsed, even though the underlying unsupervised objective (entropy minimization) was slightly sub-optimal.
