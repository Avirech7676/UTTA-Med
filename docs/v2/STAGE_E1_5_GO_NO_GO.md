# Stage E1.5: Final Readiness Determination & Decision Gate

**Auditor & Lead Investigator:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Determination:** **CONDITIONAL GO for Stage E2 Target Evaluation**  
**Condition:** Evaluation must execute under the reconciled configuration `configs/v2/safeadapt_e2_frozen.yaml` rather than the muted `safeadapt_e1_frozen.yaml` detector setting.  
**Target Domain:** Camelyon17 Hospital 2 (Center 2, Test WSIs `[20..29]`) — STRICTLY UNACCESSED DURING E1.5

---

## 1. Executive Summary & Audit Findings

Stage E1.5 was commissioned to conduct an exhaustive scientific readiness audit prior to unsealing the Hospital 2 target cohort. This audit produced four major determinations:

1. **Detection Defect Identified & Documented:** In Stage E1, the collapse monitor omitted `min_samples_for_alarm`, causing it to default to 100 patches while `window_size` was capped at 10. Consequently, `is_sufficient_samples` was permanently `False` across all batches, meaning the detector never evaluated an alarm decision. The reported "0 false alarms" reflected detector muting rather than low false-alarm frequency.
2. **Corrected Configuration Prespecified:** The reconciled target configuration ([safeadapt_e2_frozen.yaml](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/configs/v2/safeadapt_e2_frozen.yaml)) restores calibrated Stage C2 parameters ($W=50, \min=20$), and the sampling budget is prespecified at 128 patches/WSI (4 batches of 32), providing multiple active monitoring windows per slide.
3. **Parameter Bound Re-interpreted:** Observed parameter displacement ($2.003 \times 10^{-5}$ vs. bound $0.05$) indicates gentle adaptation, not an active containment of an explosive divergence.
4. **Stress-Test Re-interpreted:** The tumor-ascending continuous stream demonstrated sequence-dependent calibration decay in an adversarial offline stress test, supporting that per-WSI resets eliminate cross-slide carryover in this specific experiment rather than proving universal clinical safety.

---

## 2. Gate Decision Matrix

| Readiness Dimension | Pre-Evaluation Invariant | Audit Finding | Gate Status |
| :--- | :--- | :--- | :---: |
| **Detector Functionality** | Monitor must evaluate valid alarm states on streaming batches. | Flawed E1 config muted detector; reconciled E2 configuration ($W=50, \min=20, N=128$) verified on CPU. | **RESOLVED via E2 CONFIG** |
| **Arm Parity** | Arms A, B, C, D must differ only in designated safety interventions. | Verified bit-exact patch selection (280/320), identical loss formulation, and identical pre-adaptation timing. | **PASS** |
| **Boundary Isolation** | Per-WSI resets must guarantee slide-order invariance. | Verified bit-exact probability invariance ($\Delta_{\text{max}} = 0.00e+00$) across reverse and ascending schedules. | **PASS** |
| **Code Reliability** | Complete test suite green without regressions. | `pytest tests/ -q` executed: **99 passed, 2 skipped, 0 failed** in 43.74s. | **PASS** |
| **Hospital 2 Isolation** | Zero access, zero dataloaders, zero inferences on Center 2. | Confirmed zero Center-2 operations throughout Stages B, C1, C2, D, E1, and E1.5. | **PASS** |

---

## 3. The Single Prespecified Condition for Stage E2

Evaluation on Hospital 2 is authorized **only under the following binding protocol**:
1. **Active Configuration:** Must execute `configs/v2/safeadapt_e2_frozen.yaml` (hash-locked).
2. **Deterministic Sampling:** Exactly 128 patches per WSI across all 10 Center-2 slides (1,280 patches total), sampled via deterministic spatial stride $s = \lfloor N / 128 \rfloor$.
3. **Timing & Reset Invariants:** Strict pre-adaptation prediction timing and atomic per-WSI boundary resets.
4. **No Post-Hoc Tuning:** The evaluation is executed once. If SafeAdapt underperforms or triggers fallbacks, the outcome will be reported faithfully as a negative finding with zero parameter tuning.

---

## 4. Final Sign-Off

The audit phase is formally concluded. Permission is granted to proceed to **Stage E2: Frozen Hospital-2 Target Evaluation**.
