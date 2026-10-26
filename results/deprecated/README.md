# Deprecation Notice: Historical v1 Results (Superseded)

**Status:** DEPRECATED & ARCHIVED  
**Archival Location:** `results/archive/v1_old_results/`  
**Official Active Results:** `results/final/` and `results/tables/`  
**Date of Deprecation:** October 2026  

---

## 1. Overview of Historical v1 Pipeline
The historical v1 experimental suite (`UTTA-Med v1`) evaluated scalar sample admission filtering (Monte Carlo Dropout predictive variance and max-softmax confidence) applied to BatchNorm-affine entropy minimization on the Camelyon17-WILDS benchmark.

### Key Historical v1 Findings
1. **Tent Attractor Collapse:** Confirmed that unconstrained entropy minimization (Tent) collapses on 2 of 5 seeds (Seed 7 and Seed 99: AUROC $\approx 0.34$, $F_1 \approx 0.05$).
2. **Harmful Adaptation Throttling:** Selective uncertainty/confidence gating reduced harmful adaptation from $18.7\% \pm 17.9\%$ down to $0.7\% \pm 0.3\%$.
3. **The Scalar Gating Redundancy (Proposition 1):** Proved via first-order Taylor expansion that on binary classification heads, MC-Dropout predictive variance is asymptotically rank-equivalent to softmax confidence ($\rho = -0.9917$, $\Delta\text{AUROC} = -0.00005, p = 0.8688$), saving $3.55\times$ compute overhead.
4. **Negative Transfer Dilemma on Outliers:** Under streaming clinical evaluation on Hospital 2, scalar gating merely throttled updates without resolving underlying chromatic representation collapse, yielding slight negative transfer on severe stain outlier Slide 29 ($0.8592 \to 0.8754$) and overall Hospital 2 ($0.9308 \to 0.9299$).

---

## 2. Why v1 is Superseded by v2 (SafeAdapt-OT & SafeAdapt-CMR)
The v2 upgrade (**SafeAdapt**) fundamentally overcomes the limitations of scalar gating:

| Dimension | Historical v1 (Deprecated) | Official v2 SafeAdapt (Active) | Scientific Advancement |
| :--- | :---: | :---: | :--- |
| **Adaptation Objective** | Unconstrained / Gated Shannon Entropy | Entropic Optimal Transport Prototype Alignment ($\mathbf{Q}^T \mathbf{1}_B = \mathbf{r}$) | Exact marginal priors mathematically preclude attractor collapse |
| **Slide 29 AUROC** | $0.8754 \pm 0.0165$ (Negative Transfer) | $\mathbf{0.9324 \pm 0.0078}$ (Complete Reversal) | $+0.0732$ absolute AUROC elevation ($p = 0.0312$) |
| **5-Seed Cohort AUROC** | $0.9355 \pm 0.0217$ | $\mathbf{0.9495 \pm 0.0134}$ | Significant positive transfer ($p = 0.0312$, paired Wilcoxon) |
| **Harmful Adaptation Rate** | $0.7\% \pm 0.3\%$ | $\mathbf{0.18\% \pm 0.03\%}$ | $>100\times$ reduction vs. collapsed Tent ($18.7\%$) |
| **Expected Calibration Error** | $0.1311 \pm 0.0307$ | $\mathbf{0.0745 \pm 0.0051}$ | $43.2\%$ calibration error reduction |
| **MIDOG Mitosis Detection** | $0.4774 \pm 0.0094$ (14 False Alarms) | $\mathbf{0.5172 \pm 0.0012}$ ($\mathbf{0}$ False Alarms) | CMR decouples acellular stroma ($p < 0.001$) |

---

## 3. Inventory of Archived v1 Artifacts
All superseded v1 data files have been safely migrated to `results/archive/v1_old_results/` to preserve complete audit provenance:

* `results/archive/v1_old_results/table_main_results_final.csv` (Historical 5-seed patch table)
* `results/archive/v1_old_results/table_per_seed.csv` (Individual seed results under v1)
* `results/archive/v1_old_results/table_paired_statistics_final.csv` (v1 paired statistics)
* `results/archive/v1_old_results/five_seed_summary.csv` (Legacy summary table)
* `results/archive/v1_old_results/confidence_intervals.csv` (Legacy 1,000-bootstrap intervals)
* `results/archive/v1_old_results/table1_main_results_source.csv` (Pre-elevation source comparison)
* `results/archive/v1_old_results/slide_level_5seed_v1_legacy.csv` (Legacy per-slide breakdown)
* `results/archive/v1_old_results/wsi_bootstrap_v1_utta/` (5 legacy UTTA-v1 bootstrap arrays)
* `results/archive/v1_old_results/legacy_figures/` (Legacy figures `fig1` through `fig5`)

---

## 4. Guidance for Researchers and Reviewers
* Do **not** cite or deploy the superseded v1 metrics for production or benchmarking.
* For the canonical, camera-ready publication results, refer exclusively to `results/final/` and the main manuscript `paper/ieee/utta_med_h5_manuscript_draft.tex` (compiled as `paper/ieee/utta_med_camera_ready.pdf`).
