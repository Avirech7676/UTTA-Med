# UTTA-Med: Venue Submission Package & Checklist

**Target Track Guidance:**
- **Primary Venues (Empirical / Reliability Focus):** ISBI, EMBC, MIDL (4–8 pages). The current five-seed reliability study, failure mode analysis, and matched-coverage controls are ideally matched to these venues.
- **Stretch Venues (Requires Phase 4 Multi-class):** MICCAI, IEEE JBHI (only if NCT-CRC-HE-100K 9-class experiment is conducted to test whether MC-Dropout diverges from confidence).
- **Out of Scope for this Paper:** MedIA, IEEE TMI, CVPR (which demand novel algorithmic foundation models or extensive multi-task benchmarks).

---

## 1. Submission Checklist Sign-off

- [x] **Anonymous Manuscript:** `paper/ieee/utta_med.tex` stripped of author names, affiliations, and identifying URLs ("Anonymous Authors", "Short title: UTTA-Med Reliability Study").
- [x] **File of Record:** `paper/ieee/utta_med.tex` (+ `paper/ieee/utta_med.bib`) is the single file of record. All duplicate manuscripts moved to `archive/paper/`.
- [x] **Core & Medical TTA Citations Included:**
  - Core TTA: Tent (ICLR 2021), EATA (ICML 2022), SAR (ICLR 2023), WILDS/Camelyon17 (ICML 2021, TMI 2019).
  - Medical TTA: CertainTTA (Inf. Fusion 2025), UAD-FM (npj Digit. Med. 2025), TT-SaD (ECCV 2024), SicTTA (MedIA 2026), MedSeg-TTA (TMI 2024), Histopath-C / LATTE (WACV 2026).
  - Required Positioning Sentence: *"Unlike CertainTTA/SicTTA (segmentation / continual), we study classification reliability under a matched-coverage design and report harm, not only AUROC."*
- [x] **Exact 4 Contribution Points Verified:**
  1. Frozen leakage-controlled classification TTA protocol on official Camelyon17 hospitals.
  2. Evidence that conservative Tent still collapses on 2/5 seeds; selective adaptation prevents collapse.
  3. Harm, coverage, ECE, WSI bootstrap CIs, and matched random/confidence/entropy controls.
  4. Empirical finding that MC-Dropout variance does not outperform confidence on binary metastasis patches.
- [x] **Forbidden String Audit Passed:**
  - [x] "first uncertainty-aware" -> 0 matches.
  - [x] "outperforms confidence" -> 0 matches.
  - [x] "significantly improves AUROC over source" -> 0 matches.
  - [x] Zero drop of seeds 7 or 99 (all 5 seeds reported in main tables and breakdown).
- [x] **Data & Code Integrity:** No checkpoint `.pt` weights in submission zip. Clean repository tree with JSON tables in `results/tables/`.

---

## 2. Reproducibility Paragraph (For Manuscript / Submission Portal)

> **Reproducibility & Leakage Controls:** All experiments use official Camelyon17-WILDS benchmark splits (`wltjr1007/Camelyon17-WILDS`), with source training on Centers $\{0,3,4\}$ (302,436 patches), threshold tuning on Center $1$ (34,904 patches), and target evaluation on Center $2$ (85,054 patches across 10 whole-slide images). Evaluation is replicated across five fixed random seeds ($\{42, 123, 2024, 7, 99\}$). Source models are trained using ResNet-18 initialized with ImageNet weights, Adam optimizer ($\mathrm{lr}=10^{-4}$), batch size 128, and early stopping on Center-1 AUROC. Test-time adaptation executes with frozen running statistics (`bn_freeze_stats`), Adam learning rate $10^{-5}$, $N_{\mathrm{MC}}=20$ stochastic forward passes, single-step gradient updates ($k=1$), and gate threshold $\tau$ calibrated to the 70th percentile of Center-1 predictive uncertainty. Center-2 target labels remain strictly untouched until final metric evaluation. All numerical results, patch predictions, and per-slide bootstrap distributions are tracked in JSON/CSV artifacts.

---

## 3. Scope & Limitations Statement

> **Limitations:** (1) The empirical findings are demonstrated on a single binary histopathology task (lymph node metastasis detection in Camelyon17); on binary tasks, predictive entropy and softmax confidence are monotonic, causing MC-Dropout variance and confidence gates to produce near-identical subsets. (2) The architecture is fixed to ResNet-18 to ensure controlled, reproducible multi-seed evaluation. (3) Target whole-slide image evaluation is constrained to the 10 test WSIs in Center 2, leading to wide bootstrap confidence intervals where AUROC overlaps with the source model.

---

## 4. Phase 1 Supplementary Artifacts Included

1. `results/tables/slide_level_5seed.csv` (Per-slide clinical unit metrics: AUROC, F1, Accuracy, Mean Probability, Burden Error, Harm Rate across 350 evaluations)
2. `figures/fig_slide_tent_collapse.png` (Slide-level Tent degradation vs selective UTTA-Med preservation)
3. `figures/fig_slide_mean_prob_auroc.png` (Dual-panel clinical MIL tumor-burden calibration MAE = 0.102 and equal-weight WSI AUROC)
4. `results/tables/coverage_risk_curves.json` (Full 5-seed coverage sweeps $c \in [0.3, 1.0]$)
5. `figures/fig_coverage_risk.png` (Five-fold risk reduction curve and harm rate comparison)
6. `results/tables/tent_collapse_case_study.md` (Slide 29 and Slide 28 logit collapse mechanistic analysis)
7. `figures/fig_seed7_99_slide_damage.png` (Distribution histogram collapse to zero on Slide 29)
