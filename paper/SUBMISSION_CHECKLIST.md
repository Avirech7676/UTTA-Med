# UTTA-Med — submission checklist

Mandatory experiments COMPLETE. Optional GPU (SAR / EATA-C / temp / C) is code-ready.

## Claims you MAY make
- [x] Cross-hospital shift is real (AUROC drop 0.070 ± 0.009).
- [x] Fair Tent collapses on 2/5 seeds; slide-level significant on 7 and 99.
- [x] MC-Dropout U tracks error on hospital 1 (r ≈ 0.40). N=20 is enough.
- [x] Gating cuts harm ~25× vs Tent and raises F1 on 5/5 seeds.
- [x] Random gating also collapses; random@UTTA coverage still fails (seed 42).
- [x] UTTA-Med ≈ confidence. Entropy ≡ confidence in binary.
- [x] Inner steps k≥5 collapse. Official k=1.
- [x] WSI 95% CIs are wide (n=10 slides).
- [x] Grad-CAM is an audit, not clinical validation.

## Claims you must NOT make
- First uncertainty-aware TTA for medical images.
- UTTA-Med beats confidence on AUROC as the headline.
- Drop seeds 7 and 99.
- Mean Tent AUROC 0.70 without the 2/5 collapse.
- Patch-level CIs, or “significant AUROC gain” from WSI intervals.
- Clinical validity.

## Optional rows (seed 42 only, after next Kaggle JSON)
- Temperature scaling ECE/Brier
- SAR, EATA-C, DLTTA vs Tent/EATA
- Camelyon17-C source degradation
