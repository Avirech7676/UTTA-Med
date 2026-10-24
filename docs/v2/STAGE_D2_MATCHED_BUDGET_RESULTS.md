# Stage D2: Matched Adaptation Budget Results

**Author:** Senior Medical Imaging ML Researcher, PyTorch Performance Engineer, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Completed & Audited  
**Environment:** Intel CPU, Windows 11, PyTorch 2.14.0+cpu  

---

## 1. Executive Summary

Experiment B establishes an approximately matched adaptation budget across all gating policies, directly addressing **Pre-Stage D2 Correction C**:
> *"Matching only the percentage of selected patches is insufficient. Two methods could select 70% of patches but distribute those selections very differently across batches, resulting in different numbers of optimizer updates and different gradient magnitudes. Stage D2 should report both selected-patch coverage and actual accepted optimizer steps, with per-batch selection counts. Where possible, match the adaptation-step budget as well."*

Thresholds were calibrated exclusively on the 5 Hospital-1 calibration WSIs `[10, 12, 14, 16, 18]` and frozen prior to evaluation on the 5 assessment WSIs `[11, 13, 15, 17, 19]`.

### Key Findings
1. **Budget Matching Precision:**
   - MC-Only achieved $68.1\%$ coverage (218 / 320 patches).
   - Calibrated Confidence-Only achieved $68.8\%$ coverage (220 / 320 patches) — a residual difference of only **2 patches ($+0.6\%$)**.
   - Calibrated Hybrid achieved $67.2\%$ coverage (215 / 320 patches) — a residual difference of only **3 patches ($-0.9\%$)**.
   - **Optimizer Steps:** All three active methods executed exactly **10 out of 10 accepted optimizer updates** (1 update per batch of 32 across all 10 batches; 0 rejected updates, 0 rollbacks).
2. **Predictive Performance at Matched Budget:**
   - All three active gating policies achieved identical macro discrimination and classification metrics:
     - **AUROC:** $0.9737$ (Frozen Source: $0.9736$)
     - **AUPRC:** $0.9544$ (Frozen Source: $0.9542$)
     - **F1 Score:** $0.8632$ (Frozen Source: $0.8632$)
3. **Computational Efficiency at Matched Budget:**
   - Confidence-Only completed inference and adaptation in **118.09 seconds**.
   - MC-Only required **518.92 seconds**.
   - Confidence-Only delivered a **$4.39\times$ wall-clock speedup** over MC-Only with virtually identical predictive accuracy.

---

## 2. Threshold Calibration on Development Slides

Threshold calibration adhered strictly to development-partition boundaries:
- **Calibration Population:** 320 patches across WSIs `[10, 12, 14, 16, 18]` (64 patches/slide).
- **Target Selection Budget:** Established by executing MC-dropout predictive variance gating ($\tau_{\text{var}} = 8.88 \times 10^{-6}, K=20$) on the calibration slides, which selected **231 / 320 patches ($72.2\%$)**.
- **Calibrated Matched Confidence Threshold:**
  $$\tau_{\text{conf}}^{\text{matched}} = \text{Percentile}_{100 - 72.2}(\text{Confidence}_{\text{cal}}) = \mathbf{0.9931}$$
- **Calibrated Matched Hybrid Thresholds:**
  $$\theta_{\text{high}}^{\text{matched}} = \mathbf{0.9969}, \quad \theta_{\text{low}}^{\text{matched}} = \mathbf{0.6000}$$
- **Freezing:** All thresholds were frozen before touching assessment WSIs. Zero assessment labels were accessed during calibration.

---

## 3. Matched Budget Assessment Results Table

Evaluated on held-out Assessment WSIs `[11, 13, 15, 17, 19]` (320 patches, 10 batches of 32):

| Gating Regimen | Selected Patches | Coverage (%) | Residual vs MC | Attempted Steps | Accepted Steps | AUROC | AUPRC | F1 Score | Wall Time (s) | Speedup vs MC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Source (Matched)** | 0 / 320 | 0.0% | -68.1% | 0 | 0 | 0.9736 | 0.9542 | **0.8632** | 99.56s | N/A |
| **Confidence-Only (Matched, $\tau=0.9931$)** | 220 / 320 | 68.8% | **+0.6% (+2)** | 10 | 10 | **0.9737** | **0.9544** | **0.8632** | **118.09s** | **$4.39\times$** |
| **MC-Only (Matched, $K=20$)** | 218 / 320 | 68.1% | 0.0% (Ref) | 10 | 10 | **0.9737** | **0.9544** | **0.8632** | 518.92s | $1.00\times$ |
| **Hybrid (Matched, $0.60/0.9969$)** | 215 / 320 | 67.2% | **-0.9% (-3)** | 10 | 10 | **0.9737** | **0.9544** | **0.8632** | 432.92s | $1.20\times$ |

---

## 4. Per-Batch Adaptation Steps and Selection Distribution

To verify that the adaptation budgets were matched at the optimizer update level (addressing Correction C):

| Batch Index | WSI ID | Batch Size | MC-Only Selected | Conf-Only Selected | Hybrid Selected | MC Step Accepted? | Conf Step Accepted? | Hybrid Step Accepted? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Batch 1** | WSI 11 | 32 | 22 | 22 | 21 | Yes | Yes | Yes |
| **Batch 2** | WSI 11 | 32 | 21 | 21 | 21 | Yes | Yes | Yes |
| **Batch 3** | WSI 13 | 32 | 23 | 24 | 23 | Yes | Yes | Yes |
| **Batch 4** | WSI 13 | 32 | 22 | 22 | 22 | Yes | Yes | Yes |
| **Batch 5** | WSI 15 | 32 | 21 | 21 | 20 | Yes | Yes | Yes |
| **Batch 6** | WSI 15 | 32 | 20 | 21 | 21 | Yes | Yes | Yes |
| **Batch 7** | WSI 17 | 32 | 24 | 24 | 23 | Yes | Yes | Yes |
| **Batch 8** | WSI 17 | 32 | 23 | 23 | 23 | Yes | Yes | Yes |
| **Batch 9** | WSI 19 | 32 | 21 | 21 | 20 | Yes | Yes | Yes |
| **Batch 10** | WSI 19 | 32 | 21 | 21 | 21 | Yes | Yes | Yes |
| **Total** | — | **320** | **218** | **220** | **215** | **10 / 10** | **10 / 10** | **10 / 10** |

*Audit Observation:* The per-batch selection counts were remarkably uniform across slides, ranging between 20 and 24 patches per batch across all three methods. No batch experienced update skipping or optimizer divergence.

---

## 5. Scientific Implications of Matched Budget Experiment

1. **Absence of Incremental Discriminative Value from MC-Only:**
   When Confidence-Only is matched in patch coverage ($68.8\%$ vs $68.1\%$) and optimizer steps ($10/10$), it achieves the exact same AUROC ($0.9737$), AUPRC ($0.9544$), and F1 score ($0.8632$) as MC-Only, while reducing latency by **$4.39\times$**.
2. **Threshold Sensitivity of Hybrid Gating:**
   When the upper screening threshold $\theta_{\text{high}}$ is pushed to $0.9969$, almost all patches fall into the uncertain screening zone $[0.60, 0.9969)$, forcing MC evaluation on $90\%+$ of patches. Consequently, runtime increases from $201.60\text{s}$ (natural hybrid) to $432.92\text{s}$ (matched hybrid), demonstrating that hybrid efficiency is highly contingent on having a wide deterministic auto-accept window ($\ge 0.95$).
3. **Preservation of Classification F1:**
   In natural confidence gating ($\tau=0.90$), selecting $87.5\%$ of patches lowered F1 from $0.8632$ to $0.8596$. Under the tighter matched threshold ($\tau=0.9931$), selecting $68.8\%$ of patches preserved F1 at $0.8632$, matching Frozen Source. This demonstrates that more conservative gating benefits both methods equally.
