# UTTA-Med — Stage H3.1-R4b: Correct Checkpoint Generation and Localization Audit Report

**Audit Timestamp:** 2026-10-16T05:42:40.264008+00:00  
**Git Revision:** `UNKNOWN_NOT_A_GIT_CHECKOUT`  
**Compute Platform:** `Linux-6.18.48+-x86_64-with-glibc2.39` | GPU: `Tesla T4` (14.56 GB)  
**Final Certified Decision:** `SOURCE_LOCALIZATION_RECOVERED`  

---

## 1. Executive Summary & Certified Decision

**Official Verdict:** `SOURCE_LOCALIZATION_RECOVERED`  
**Auditor Determination:** Detectors successfully predict microscopic cellular-scale bounding boxes (mean widths: [49.64, 45.56, 49.85] px) with verified positive recall (Recalls: [0.6667, 0.5833, 0.6111]).

| Gate / Requirement | Status | Certified Values |
|---|---|---|
| Gate 1: Checkpoint Provenance | PASS | 3/3 Certified New Generation SHA-256 |
| Gate 2: Dataset Reconciliation | PASS | MIDOG.json Schema Validated |
| Gate 3: Source Split Disjointness | PASS | 20 Held-Out Validation Cases (Cases 101–150 Preserved) |
| Gate 4: Frozen 300-Window Manifest | PASS | Frozen Non-overlapping 512x512 Tiling |
| Gate 5: Ground-Truth Geometry | PASS | 36 unique mitoses (Mean 50.0x50.0 px) |
| Gate 6: Proposal Geometry Analysis | PASS | Bounding-Box Area Ratio Deconstructed |
| Gate 7: Centroid Euclidean Matching | PASS | Radius = 7.5 µm (30 px) Evaluated across 6 Thresholds |
| Gate 8: Visual Overlay Verification | PASS | Real Tissue Overlays Rendered for Seeds 42, 7, 123 |
| Gate 9: Executable Assertions Suite | PASS | 12/12 Consistency Checks Passed |

---

## 2. Checkpoint Provenance & Inventory

| Seed | Filename | Expected SHA-256 | Actual SHA-256 | Status |
|---|---|---|---|---|
| 42 | `source_fasterrcnn_s42.pt` | `af9c34437e37cb13...` | `af9c34437e37cb13...` | **AUTHENTICATED** |
| 7 | `source_fasterrcnn_s7.pt` | `3f1ed8e0e3997eac...` | `3f1ed8e0e3997eac...` | **AUTHENTICATED** |
| 123 | `source_fasterrcnn_s123.pt` | `44584d70480a7913...` | `44584d70480a7913...` | **AUTHENTICATED** |

---

## 3. Bounding-Box Geometry: Predicted Proposals vs Ground-Truth Mitoses

| Metric | Ground Truth Mitoses | Seed 42 Predictions | Seed 7 Predictions | Seed 123 Predictions |
|---|---|---|---|---|
| **Mean Width (px)** | 50.0 | 49.6 | 45.6 | 49.9 |
| **Mean Height (px)** | 50.0 | 20.5 | 47.2 | 44.9 |
| **Mean Area (px²)** | 2500 | 975 | 2178 | 2245 |
| **Frac Boxes > 25% Tile Area** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Whole-Patch Collapse Flag** | Cellular | RECOVERED | RECOVERED | RECOVERED |

---

## 4. Centroid Localization Performance (at Prespecified τ = 0.10)

| Detector Seed | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| Seed 42 | 24 | 318 | 12 | 0.0702 | 0.6667 | **0.1270** |
| Seed 7 | 21 | 51 | 15 | 0.2917 | 0.5833 | **0.3889** |
| Seed 123 | 22 | 37 | 14 | 0.3729 | 0.6111 | **0.4632** |

---

## 5. Threshold Sensitivity Sweeps

| Seed | τ = 0.05 (F1 / Rec) | τ = 0.10 (F1 / Rec) | τ = 0.20 (F1 / Rec) | τ = 0.30 (F1 / Rec) | τ = 0.40 (F1 / Rec) | τ = 0.50 (F1 / Rec) |
|---|---|---|---|---|---|---|
| Seed 42 | 0.125 / 0.722 | 0.127 / 0.667 | 0.210 / 0.611 | 0.202 / 0.528 | 0.208 / 0.528 | 0.375 / 0.500 |
| Seed 7 | 0.346 / 0.639 | 0.389 / 0.583 | 0.442 / 0.528 | 0.444 / 0.444 | 0.457 / 0.444 | 0.478 / 0.444 |
| Seed 123 | 0.358 / 0.611 | 0.463 / 0.611 | 0.500 / 0.528 | 0.514 / 0.500 | 0.531 / 0.472 | 0.475 / 0.389 |

---

## 6. Audit Conclusion & Protocols

**Audit Determination:** Detectors successfully predict microscopic cellular-scale bounding boxes (mean widths: [49.64, 45.56, 49.85] px) with verified positive recall (Recalls: [0.6667, 0.5833, 0.6111]).

**Status: UNLOCKED FOR STAGE H3.2 TARGET ADAPTATION.**
