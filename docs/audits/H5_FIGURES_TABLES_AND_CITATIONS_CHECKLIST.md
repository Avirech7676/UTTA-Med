# UTTA-Med — Stage H5: Figures, Tables, and Citations Checklist (Revised)

**Document Reference:** `UTTA-MED-H5-FIGURES-TABLES-CITATIONS-CHECKLIST-R1`  
**Target Manuscript Draft:** [`paper/ieee/utta_med_h5_manuscript_draft.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/paper/ieee/utta_med_h5_manuscript_draft.tex)  
**Lead Auditor:** Senior Computational Pathology Researcher & Technical Editor  
**Date:** October 18, 2026 (Revision 1 Post-Review)  
**Status:** COMPLETE & VERIFIED

---

## 1. Manuscript Figures Inventory & Physical Image Audit

All figures referenced in the main text are physically present in `paper/ieee/figs/`, verified for resolution, format, and scientific caption accuracy:

| Figure Key | Physical File Path | Dimensions / Type | Caption & Content Verification | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Fig. 1 (`fig:d5`)** | `figs/fig_d5_coverage_risk.png` | PNG (77.7 KB) | **Coverage--risk (reject option) curve on Camelyon17 Hospital~2.** Confirms that retaining the 50% most confident patches reduces source error from 16.6% to 3.6% ($>4.5\times$ reduction). | **PASS** |
| **Fig. 2 (`fig:d6`)** | `figs/fig_d6_slide29_seed7_hist.png` | PNG (39.8 KB) | **Seed 7, slide 29 prediction histograms.** Contrasts Tent collapse driving probability mass toward $p \approx 0$ against selective gating preserving separation. | **PASS** |
| **Fig. 3 (`fig:d2`)** | `figs/fig_d2_tumor_burden.png` | PNG (66.8 KB) | **Tumor-burden calibration on Hospital~2 WSIs (source model).** Scatter plot of mean predicted probability vs true tumor fraction per slide. Caption explicitly cautions: *"Not a MIL AUROC."* | **PASS** |

### Supplementary & Extended Figure Pool in `paper/ieee/figs/`
The repository maintains an additional 28 generated figures available for journal extension or supplementary inspection:
- `fig_architecture.png` (System pipeline and selective gating controller flow)
- `fig_cam_frozen_lowu.png`, `fig_cam_frozen_highu.png`, `fig_cam_lowu.png`, `fig_cam_highu.png` (Grad-CAM interpretability maps)
- `fig_cam_fn.png`, `fig_cam_fp.png` (Confident error failure mode visualizations)
- `fig_coverage_risk_tradeoff.png`, `fig_harm.png`, `fig_ksteps.png` (Sensitivity sweeps)
- `fig_main_auroc.png`, `fig_main_bars.png`, `fig_per_seed.png`, `fig_reliability.png` (Evaluation plots)
- `fig_seed7_99_slide_damage.png`, `fig_slide_tent_collapse.png`, `fig_tent_collapse_mechanism.png` (Collapse forensics)
- `fig_umap.png` (Latent feature representations under shift)

---

## 2. Manuscript Tables Inventory & Cross-Check (Revised)

| Table Key | Caption | Data Source | Revision Changes |
| :--- | :--- | :--- | :--- |
| **Table I (`tab:main`)** | Camelyon17-WILDS Target Hospital 2 ($N=85{,}054$ patches). Mean $\pm$ SD over five random seeds. | `paper/tables/table2_five_seed_primary.csv` | Maintained; confirmed 100% exact match. |
| **Table II (`tab:perseed`)** | Per-seed target AUROC on Hospital 2. Collapse cells highlighted in bold. | `paper/tables/table2_per_seed_auroc.csv` | Maintained; confirmed 100% exact match. |
| **Table III (`tab:h4_metrics`)** | Held-Out MIDOG Scanner 3 Detection Performance Across Five Adaptation Arms and Three Random Seeds ($r=30.0$ px, $\tau=0.20$, $h \ge 1.0$ px, 50 Cases, 1,100 Patches, 118 Ground-Truth Mitoses). | `results/generalization/h4/phase_b/h4_per_arm_metrics.csv` | **Updated fallback timing:** Arm E fallback latched at patch 102 (Seed 42), patch 80 (Seed 7), patch 58 (Seed 123). |
| **Table IV (`tab:h4_bootstrap`)** | Paired Case-Clustered Bootstrap Contrasts ($B=10{,}000$ Resamples) on Scanner 3. | `results/generalization/h4/phase_b/h4_paired_bootstrap_contrasts.csv` | **Updated determination:** Replaced "Equivalent ($\pm 0.03$)" with "95% bootstrap CI within $\pm 0.03$; descriptive". |

---

## 3. Citations & Bibliography Reconciliation (Revised)

The bibliography file `paper/ieee/utta_med_h5.bib` contains 22 total references and is positioned after the appendices in accordance with IEEE formatting conventions:

| Citation Key | Type | Corrected Authors & Title | Venue / Year / Identifier | Status |
| :--- | :---: | :--- | :---: | :---: |
| `wang2021tent` | Inproceedings | Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization* | ICLR 2021 | **VERIFIED** |
| `niu2022eata` | Inproceedings | Niu et al., *Efficient Test-Time Model Adaptation without Forgetting* | ICML 2022 | **VERIFIED** |
| `niu2023sar` | Inproceedings | Niu et al., *Towards Stable Test-Time Adaptation in Dynamic Wild World* | ICLR 2023 | **VERIFIED** |
| `koh2021wilds` | Inproceedings | Koh et al., *WILDS: A Benchmark of in-the-Wild Distribution Shifts* | ICML 2021 | **VERIFIED** |
| `bandi2018camelyon17` | Article | Bandi et al., *From Detection of Individual Metastases to Classification...* | IEEE TMI 2019 (online 2018) | **CORRECTED** |
| `gal2016dropout` | Inproceedings | Gal & Ghahramani, *Dropout as a Bayesian Approximation...* | ICML 2016 | **VERIFIED** |
| `he2016resnet` | Inproceedings | He et al., *Deep Residual Learning for Image Recognition* | CVPR 2016 | **VERIFIED** |
| `selvaraju2017gradcam`| Inproceedings | Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks...* | ICCV 2017 | **VERIFIED** |
| `guo2017calibration` | Inproceedings | Guo et al., *On Calibration of Modern Neural Networks* | ICML 2017 | **VERIFIED** |
| `yang2023medmnist` | Article | Yang et al., *MedMNIST v2: A Large-Scale Lightweight Benchmark...* | Sci. Data 2023 | **VERIFIED** |
| `hendrycks2019benchmarking` | Inproceedings | Hendrycks & Dietterich, *Benchmarking Neural Network Robustness...* | ICLR 2019 | **VERIFIED** |
| `liang2023comprehensive` | Article | Liang et al., *A Comprehensive Survey on Test-Time Adaptation...* | arXiv 2023 | **VERIFIED** |
| `certaintta2025` | Article | Dong et al., *CertainTTA: Estimating Uncertainty for Test-Time...* | Inf. Fusion 2025 | **VERIFIED** |
| `uadfm2025` | Article | Lou et al., *Uncertainty-Aware and Causal Test-Time Adaptive Foundation...* | npj Digit. Med. 2025 | **VERIFIED** |
| `ttsad2024` | Inproceedings | Tsai et al., *Test-Time Stain Adaptation with Diffusion Models...* | ECCV 2024 | **VERIFIED** |
| `sictta2026` | Article | Wu et al., *SicTTA: Single Image Continual Test Time Adaptation...* | Med. Image Anal. 2026 | **VERIFIED** |
| `histopathc2026` | Inproceedings | Noori et al., *Histopath-C: Towards Realistic Domain Shifts...* | WACV 2026 | **VERIFIED** |
| `lee2024aetta` | Inproceedings | Lee et al., *AETTA: Label-Free Accuracy Estimation for Test-Time...* | CVPR 2024 | **VERIFIED** |
| `bahumanya2026explanationstability` | Misc | Bahumanya et al., *Explanation Stability of Test-Time Adaptation...* | arXiv 2026 | **VERIFIED** |
| `yu2025medsegtta` | Misc | Yu et al., *A Large Scale Benchmark for Test Time Adaptation...* | arXiv 2025 | **VERIFIED** |
| `aubreville2023mitosis` | Article | Aubreville et al., *Mitosis Domain Generalization in Histopathology Images --- The MIDOG Challenge* | Medical Image Analysis 2023<br>DOI: 10.1016/j.media.2022.102699 | **CORRECTED** |
| `ren2015fasterrcnn` | Inproceedings | Ren et al., *Faster R-CNN: Towards Real-Time Object Detection...* | NeurIPS 2015 | **VERIFIED** |

---

## 4. Verification Summary
All figures, tables, cross-references, and bibliographic citations are verified, mutually consistent, free of literal LaTeX underscores, and structurally aligned with IEEE standards.
