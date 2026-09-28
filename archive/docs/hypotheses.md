# Hypotheses — UTTA-Med

**Frozen before any target-test evaluation.**

- **H1**: Cross-hospital distribution shift reduces source-only performance (AUROC, AUPRC, F1, sensitivity, specificity).
- **H2**: Standard TTA (Tent and EATA) improves target performance relative to no adaptation.
- **H3**: Higher MC-Dropout predictive uncertainty is associated with higher prediction error on the target domain.
- **H4**: Uncertainty-gated TTA reduces the harmful adaptation rate (Correct → Wrong) compared with unrestricted TTA.
- **H5**: UTTA-Med improves (or at least does not degrade) calibration (lower ECE / Brier) relative to unrestricted TTA.
- **H6**: The uncertainty gate produces a meaningful performance–coverage trade-off that is superior to Random / Confidence / Entropy gates under matched coverage.
- **H7**: Grad-CAM can reveal qualitative differences in model attention after adaptation (tissue vs. artifact / background).

All hypotheses are tested with multi-seed evaluation (≥5 seeds) and WSI-level bootstrap confidence intervals.
