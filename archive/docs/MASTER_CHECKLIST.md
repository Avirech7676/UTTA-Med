# UTTA-Med Master Plan v4 — live checklist

Mandatory 5-seed science is **locked**. Optional seed-42 TTA rows are **signed off** (with two caveats below). Paper is **last**.

## Phase-gate (§56)

| # | Gate | Status |
|---|---|---|
| 1 | PathMNIST pipeline | SKIP |
| 2–20 | MC-Dropout through WSI bootstrap 5 seeds | **DONE** |
| 21 | Robustness (Camelyon17-C) | **INVALID run** (all-negative 8k prefix). Shuffle fix optional. |
| 22 | Reproducibility audit | DONE |
| 23 | Manuscript | LAST |

## Optional skip-list (quoted) — seed 42 only

| Item | Status |
|---|---|
| Temperature scaling | **DONE** T=1.87 on hospital 1; ECE 0.099→0.069; AUROC unchanged |
| SAR | **DONE** AUROC 0.948, F1 0.828, harm 0.054, cov 0.93, 116 resets |
| EATA-C | **RAN, not usable** coverage 0.001 — diversity filter empty. Do not retune on test. |
| DLTTA | **DONE** AUROC 0.952, F1 0.876, harm 0.024 (seed 42 only; not Table 1) |
| Camelyon17-C | **RAN, invalid** first 8k patches all class 0 |
| ResNet-50 / PathMNIST / Camelyon16 / MIDOG++ / ViT / 10 seeds | SKIP |

## Frozen 5-seed claims (unchanged)

Gating cuts harm ~25× and raises F1 on 5/5. UTTA-Med ≈ confidence. Tent collapses 2/5. Do not put SAR/DLTTA in Table 1.

## Next

Optional 15 min: shuffled Camelyon17-C (`kaggle_camelyon17c_shuffle.py`).  
Else: **documents**.
