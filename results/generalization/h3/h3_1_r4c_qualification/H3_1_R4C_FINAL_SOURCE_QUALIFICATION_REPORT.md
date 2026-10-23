# UTTA-Med — Stage H3.1-R4c: Final Source Detector Qualification Report

**Execution Timestamp:** 2026-10-16T03:28:04.199584+00:00  
**Compute Platform:** Tesla T4 (14.56 GB)  
**Git Revision:** `UNKNOWN_NOT_A_GIT_CHECKOUT`  
**Qualification Status:** `QUALIFICATION_AUDIT_COMPLETED_WITH_KNOWN_LIMITATIONS`  

## 1. Executive Summary & Measured Findings

Stage H3.1-R4c establishes definitive empirical qualification of the three certified source detectors (Seeds 42, 7, 123). In accordance with scientific audit protocols, this report refrains from self-certifying readiness and reports exact measured metrics alongside confirmed model pathologies.

### Primary Performance Matrix (at Operating Threshold $\tau = 0.10$)

| Seed | Evaluation Mode | TP | FP | FN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 42 | `raw_unfiltered` | 81 | 959 | 22 | 0.0779 | 0.7864 | **0.1417** |
| 42 | `boundary_filtered_h1` | 81 | 274 | 22 | 0.2282 | 0.7864 | **0.3537** |
| 7 | `raw_unfiltered` | 74 | 183 | 29 | 0.2879 | 0.7184 | **0.4111** |
| 7 | `boundary_filtered_h1` | 74 | 183 | 29 | 0.2879 | 0.7184 | **0.4111** |
| 123 | `raw_unfiltered` | 71 | 120 | 32 | 0.3717 | 0.6893 | **0.4830** |
| 123 | `boundary_filtered_h1` | 71 | 120 | 32 | 0.3717 | 0.6893 | **0.4830** |

---

## 2. Task 1: Checkpoint Provenance Audit

| Seed | Checkpoint Filename | Expected SHA-256 | Actual Verified SHA-256 | Provenance Status |
|---|---|---|---|---|
| 42 | `source_fasterrcnn_s42.pt` | `af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003` | `af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003` | **AUTHENTICATED** |
| 7 | `source_fasterrcnn_s7.pt` | `3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e` | `3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e` | **AUTHENTICATED** |
| 123 | `source_fasterrcnn_s123.pt` | `44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348` | `44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348` | **AUTHENTICATED** |

### Provenance Lineage Confirmation
- **Training Source Code:** `notebooks/official/kaggle_midog_source_train_repaired.py`
- **Execution Timestamp:** 2026-10-15T03:27:00Z to 2026-10-15T03:43:00Z
- **Training Hardware:** Kaggle Tesla T4 GPU (Linux 6.18, PyTorch 2.11.0+cu128)
- **Training Split:** 80 Training cases (Scanners 1 & 2), 20 Validation cases (Disjoint). Scanner 3 Strictly Excluded.
- **Lineage Separation:** The verified generation (`af9c3443...`, `3f1ed8e0...`, `44584d70...`) was initialized from standard torchvision ImageNet pre-trained ResNet-50 weights. It has ZERO connection or weight inheritance with earlier draft hashes (`f3651669`, `bf0f9df2`, `6da69214`).

---

## 3. Task 2: Seed-42 Bounding-Box Pathology Root Cause

- **Total Predictions (at $\tau = 0.10$):** 1040
- **Sub-1px Height Predictions:** 685 (65.87%)
- **Bottom-Boundary Pinned ($y_2 \ge 511.9$ px):** 685 (100.0% of sub-1px boxes)
- **Interior Sub-1px Predictions:** 0 (Zero interior degeneracies)

### Root-Cause Dissection
1. **Not Coordinate Conversion or Preprocessing:** Input transforms and coordinate scales are standard 512x512 RGB.
2. **Not Model Regression Collapse:** When unclipped interior boxes are evaluated, Seed 42 outputs cellular proposals (mean width 47.5 px, mean height 48.2 px), matching ground truth.
3. **Mechanism:** Faster R-CNN's Region Proposal Network (RPN) retains an active boundary anchor at the bottom edge ($y \approx 536$ px). The RoI box predictor assigns marginal foreground confidence ($s \approx 0.15 - 0.47$). When torchvision's `clip_boxes_to_image` clips the box to the $512 \times 512$ tile boundary, $y_2$ is truncated to $512.0$, leaving a 0.1 px high horizontal slit pinned to the edge.
4. **Impact on Detection:** Applying a standard post-clipping filter ($h \ge 1.0$ px) suppresses all 685 spurious slit false positives while retaining 100% of True Positives (81/81), increasing precision from 0.0779 to 0.2282 and F1 from 0.1417 to 0.3537.

---

## 4. Task 3: Expanded Source-Validation Coverage

- **Evaluated Windows:** 900 non-overlapping $512 \times 512$ tissue tiles across 20 held-out cases.
- **Total Evaluated Tissue Area:** 14.75 mm² (62.2 HPF equivalents).
- **Ground-Truth Mitoses Count:** 103 mitoses (Objective >= 100 achieved).
- **Positive Windows:** 83 | **Negative Windows:** 817.

---

## 5. Unresolved Limitations & Auditor Notes

1. **Seed-42 Boundary Sensitivity:** Seed 42's RPN emits boundary proposals that degenerate upon edge clipping. While cellular recall is intact, automated evaluation without a post-clipping dimension filter depresses precision.
2. **Threshold Dependency:** Detector precision is sensitive to confidence threshold $\tau$; peak F1 occurs between $\tau = 0.20$ and $\tau = 0.40$.
3. **Strict Scientific Prudence:** In accordance with audit instructions, scientific readiness is not self-certified. The empirical evidence is archived for independent peer audit.
