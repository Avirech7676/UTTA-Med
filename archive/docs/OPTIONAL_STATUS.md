# Status of Supplemental & Optional Experiments

This document records the exact status of optional, exploratory, and historical runs to prevent misattribution or citing incomplete artifacts.

---

### 1. Headline vs. WSI Rerun: Seed-42 EATA Note

- **Headline 5-Seed Table (Table 1 / Table 2)**:
  Uses the frozen point-estimate: **AUROC = 0.9363** (Harm = 0.0514, Coverage = 0.9400).
- **WSI Bootstrap Slide-Level Rerun**:
  Yields point estimate: **AUROC = 0.9386** (Harm = 0.0112).
- **Methodological Guidance**:
  Both numbers are valid and documented in `PHASE14_FINAL_AUDIT.md`. The 5-seed pooled comparison preserves the original locked checkpoint run (`0.9363`), while the 1000-resample slide bootstrap uses the slide-resampled array. Do not mix point estimates across contexts.

---

### 2. Invalid or Unusable Supplemental Artifacts

| Experiment | Status | Reason & Guidance |
| :--- | :---: | :--- |
| **Camelyon17-C** | **INVALID — DO NOT CITE** | First 8,000 samples were all-negative due to sequential chunk ordering prior to shuffle fix. |
| **EATA-C** | **UNUSABLE OPTIONAL** | Filter criterion produced extremely low coverage ($\approx 0.001$) under frozen medical thresholds; not comparable. |

---

### 3. Supplemental Seed-42 Only Methods

The following methods were evaluated solely on Seed 42 as exploratory baselines and are **not** part of the authoritative 5-seed primary table:
- **SAR (Selective Adaptation for Robustness)**: Seed 42 only (`results/optional/camelyon17_remaining_optional_s42.json`).
- **DLTTA (Dynamic Label Test-Time Adaptation)**: Seed 42 only.
- **Post-Hoc Temperature Scaling ($T=1.87$)**: Fitted on Center 1 validation; evaluated on Center 2 (AUROC unchanged at 0.9355, ECE reduced from 0.0988 to 0.0691).
