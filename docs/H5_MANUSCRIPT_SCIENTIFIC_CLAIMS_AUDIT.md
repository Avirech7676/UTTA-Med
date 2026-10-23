# UTTA-Med — Stage H5: Manuscript Scientific Claims Audit (Revised)

**Document Reference:** `UTTA-MED-H5-SCIENTIFIC-CLAIMS-AUDIT-R1`  
**Target Manuscript Draft:** [`paper/ieee/utta_med_h5_manuscript_draft.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/paper/ieee/utta_med_h5_manuscript_draft.tex)  
**Lead Auditor:** Senior Computational Pathology Researcher & Independent Scientific Auditor  
**Date of Audit:** October 18, 2026 (Revision 1 Post-Review)  
**Audit Scope:** Verification of empirical claims, methodological constants, fallback timing, statistical equivalence phrasing, and cryptographic provenance following independent review.

---

## Executive Summary & Review Reconciliations

Following formal independent review, this revised audit reconciles all methodological parameters, algorithm constants, fallback timelines, and statistical claims against verified runner code and execution logs:

| Item Under Review | Initial Manuscript Claim | Verified Empirical & Code Truth | Revised Manuscript Disposition | Audit Status |
| :--- | :--- | :--- | :--- | :---: |
| **Detection Proposal Admission** | Arm B uses $p \ge 0.25$ | H4 runner uses $\tau = 0.20$ (`PRIMARY_OPERATING_THRESHOLD`) | Corrected to $\tau = 0.20$ in Section III-C | **VERIFIED** |
| **Absolute Proliferation Alarm** | Triggers above 15 detections | H4 runner triggers above 4 (`CALIBRATED_PROLIFERATION_MAX_PER_PATCH = 4`) | Corrected to $n_{\mathrm{pred}} > 4$ in Section III-C | **VERIFIED** |
| **BatchNorm Promotion** | "Converts FrozenBatchNorm into nn.BatchNorm2d" | Promotes `FrozenBatchNorm2d` scale $\gamma$ and bias $\beta$ buffers to trainable parameters | Corrected architecture description in Section III-A | **VERIFIED** |
| **Runner Provenance** | Attributed hash `0a9d69af...` to launcher | Launcher hash is `d2c15edc...`; executed runner is `0a9d69af...` | Separated launcher and staged runner in Section VI-B | **VERIFIED** |
| **Fallback Timing (Seed 42)** | Global patch 89 | Global patch 102 (36 updates, 67 skipped updates; predictions differ up to patch 102) | Reconciled to global patch 102 in Section VI-B | **VERIFIED** |
| **Fallback Timing (Seed 7)** | Global patch 48 | Global patch 80 (2 updates, 79 skipped updates; silence across patches 49--80) | Reconciled to global patch 80 in Section VI-B | **VERIFIED** |
| **Fallback Timing (Seed 123)** | Global patch 48 | Global patch 58 (2 updates, 57 skipped updates; silence across patches 27--58) | Reconciled to global patch 58 in Section VI-B | **VERIFIED** |
| **Statistical Equivalence** | Labeled Arm E vs A as "Equivalent ($\pm 0.03$)" | All-zero bootstrap differences under fixed stream do not prove clinical equivalence | Replaced with "95% bootstrap CI within $\pm 0.03$; descriptive" | **VERIFIED** |
| **MC-Dropout vs Confidence** | "Rank identical patches" | Monotone alignment in binary classification, but patch rankings not directly correlated | Replaced with "achieved similar observed performance" | **VERIFIED** |
| **Matched Coverage Control** | Claimed pure selection quality isolation | Realized coverage differed (70% random vs 58% gated) | Qualified: does not isolate quality from volume without caveat | **VERIFIED** |
| **Classification Uncertainty** | 5 seeds presented without cohort caveat | 5 seeds capture optimization stochasticity on same patient cohort | Explicitly noted distinction from hospital-level generalization | **VERIFIED** |
| **Limitations Section** | "Exploratory Classification" | Classification was preregistered; Detection had the threshold amendment | Corrected heading to "Exploratory Detection" | **VERIFIED** |
| **Compute Environments** | Undifferentiated hardware mention | Camelyon17 (V100/RTX 3090); MIDOG (Kaggle Tesla T4) | Distinguish compute platforms in Limitations | **VERIFIED** |
| **Bibliography Entry MIDOG** | Cited Nature Sci. Data 2023 dataset paper | Central challenge paper is Aubreville et al., *Med. Image Anal.* 2023 | Corrected bib entry with DOI `10.1016/j.media.2022.102699` | **VERIFIED** |
| **Bibliography Entry CAMELYON17** | Year 2018 | Appeared in print in IEEE TMI 2019 (online 2018) | Corrected year to 2019 with note | **VERIFIED** |
| **LaTeX Syntax & Layout** | Literal underscores in text; bib before appendix | Underscores in text cause math errors; bib belongs after appendix | Escaped underscores with `\texttt{...}`; moved bib after appendix | **VERIFIED** |

---

## 1. Five-Point Conceptual Taxonomy Audit

### 1. Accuracy Improvement
- **Camelyon17 Metastasis Classification:** Demonstrates statistically significant $F_1$ improvement across all five seeds: $\Delta F_1 = +0.0231$ (95% CI $[+0.0104, +0.0358]$, $t=5.053$, $p=0.0072$, Holm-adjusted $p \approx 0.05$). AUROC improvement is $+0.0058$ (95% CI $[-0.0121, +0.0237]$, $t=0.887$, $p=0.4199$, non-significant).
- **MIDOG Scanner 3 Mitotic Detection:** Demonstrates **zero** accuracy improvement over frozen baseline: $\Delta F_1 = 0.0000$ (95% CI $[0.0000, 0.0000]$).
- *Audit Status:* **PASS.** The manuscript explicitly states that SafeAdapt achieved zero accuracy gain on Scanner 3 and does not describe detection results as superior.

### 2. Failure Containment Through Fallback
- **Camelyon17 Metastasis Classification:** Hard uncertainty/confidence gating prevents Tent collapse (which occurred on Seeds 7 and 99, dropping AUROC to $\approx 0.34$), reducing harm from $0.187 \pm 0.179$ to $0.007 \pm 0.003$ ($>25\times$ reduction).
- **MIDOG Scanner 3 Mitotic Detection:** SafeAdapt functions as a fail-closed circuit breaker. Upon triggering three consecutive detection silence alarms, the controller executed latching global fallback to the certified source state, preventing parameter drift.
- *Audit Status:* **PASS.** Framed strictly as an operational safeguard and failure containment mechanism, not an accuracy enhancer.

### 3. Uncertainty in Statistical Comparisons
- **Camelyon17:** 95% confidence intervals on AUROC $[-0.0121, +0.0237]$ cross zero. Five seeds capture training stochasticity on a single target center (10 WSIs), distinct from patient-level or multicenter generalization.
- **MIDOG Scanner 3:** Paired case-clustered bootstrap intervals ($B=10{,}000$ resamples) for Conventional TTA (Arm B: $[-0.0417, +0.0255]$) and Monitored TTA (Arm D: $[-0.0412, +0.0262]$) cross zero, establishing insufficient evidence of statistically reliable degradation or SafeAdapt advantage.
- *Audit Status:* **PASS.** Full confidence intervals reported; no claim of statistical superiority or rejection of the null hypothesis on detection contrasts.

### 4. Computational Correctness
- **Pre-Launch Frozen Runner:** SHA-256 `e2c8bb896b50d54e9b6118c064e043615d91d3568e1b580afbf82e9353103da6`.
- **Launcher Script:** SHA-256 `d2c15edc6ee0569be8812aa35c9ac758ba4b1922b69ebbbec5f66cd16a02334b`.
- **Executed Staged Runner:** SHA-256 `0a9d69afd9e33e3eb3a1ca8750436a25a67fb7ff8f2849277560bd20a42d6dab`.
- **Frozen Manifest:** SHA-256 `e5a8345a68ba7f0c15162c0865ae37aa9730fcc7051525abfa9ff9f5ca08d1c8` (1,100 patches).
- **Evaluation Ground Truth:** SHA-256 `fc33d0e3904e204409f18fe41f841308eb4a81d0c79f662e0ab8f6997082c612` (118 mitoses).
- *Audit Status:* **PASS.** All hashes verified independently.

### 5. Clinical Validation
- **Status:** **NOT ESTABLISHED.**
- *Audit Status:* **PASS.** Abstract, Introduction, Discussion, Limitations, and Conclusion explicitly state that clinical validation has not been performed (no multi-reader diagnostic studies, no patient outcome trials, no prospective validation).

---

## 2. Reconciled Fallback Timing Evidence Ledger

The fallback patch indices were reconciled from `results/generalization/h4/phase_b/h4_per_arm_metrics.csv`, `h4_raw_predictions.csv`, and `kaggle_midog_target_adaptation_h4_executed.py`:

```
Accounting Rule:
  In Arm E, every patch evaluated before latching global fallback increments either:
    - controller.total_updates (if admitted and updated)
    - controller.skipped_updates (if alarm tripped or 0 proposals admitted)
  Once latching global fallback is engaged (in_global_fallback = True):
    - line 1002 executes `continue`
    - neither counter advances for any subsequent patch
  Therefore, total active adaptation patches = updates + skipped_updates.
  Zero-based global patch index of fallback latching = updates + skipped_updates - 1.
```

### Empirical Traceability
1. **Seed 42:**
   - Metrics: `updates = 36`, `skipped_updates = 67`. Total = 103 patches.
   - Fallback Index: Global patch **102** ($103 - 1$).
   - Prediction Verification: In `h4_raw_predictions.csv`, Arm E differs from Arm A on 32 patches between patch 29 and patch 102. From patch 103 onward, all predictions are 100% bitwise identical to Arm A.
2. **Seed 7:**
   - Metrics: `updates = 2`, `skipped_updates = 79`. Total = 81 patches.
   - Fallback Index: Global patch **80** ($81 - 1$).
   - Silence Verification: Patches 49 through 80 had 0 detections, accumulating consecutive silence alarms that latched global fallback at patch 80. From patch 81 onward, Arm E is permanently latched to source.
3. **Seed 123:**
   - Metrics: `updates = 2`, `skipped_updates = 57`. Total = 59 patches.
   - Fallback Index: Global patch **58** ($59 - 1$).
   - Silence Verification: Patches 27 through 58 had 0 detections, accumulating consecutive silence alarms that latched global fallback at patch 58. From patch 59 onward, Arm E is permanently latched to source.

---

## 3. Algorithm Constants Reconciliation

| Constant | Description | Value in Runner | Location in Runner | Manuscript Text |
| :--- | :--- | :---: | :--- | :--- |
| `PRIMARY_OPERATING_THRESHOLD` | Detection confidence threshold for proposal admission & counting | `0.20` | Line 101, 527, 1029 | Corrected to $\tau = 0.20$ |
| `CALIBRATED_PROLIFERATION_MAX_PER_PATCH` | Absolute proliferation alarm threshold per tile | `4` | Line 106, 493 | Corrected to $n_{\mathrm{pred}} > 4$ |
| `CALIBRATED_SILENCE_WINDOW` | Silence monitoring window length | `30` | Line 104, 492 | Confirmed at $W_{\mathrm{silence}} = 30$ |
| `CALIBRATED_PROLIFERATION_SOURCE_RATIO` | Relative proliferation ratio vs source detector | `3.0` | Line 105, 494 | Confirmed at $3.0\times$ |
| `PRIMARY_MATCHING_RADIUS_PX` | Primary centroid matching distance | `30.0` | Line 99, 1269 | Confirmed at $30.0\text{ px}$ |
| `SENSITIVITY_MATCHING_RADIUS_PX` | Sensitivity centroid matching distance | `25.0` | Line 100, 1270 | Confirmed at $25.0\text{ px}$ |

---

## 4. Final Scientific Certification

All methodological statements, algorithmic constants, runner hashes, fallback indices, and statistical claims in [`paper/ieee/utta_med_h5_manuscript_draft.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/paper/ieee/utta_med_h5_manuscript_draft.tex) are 100% reconciled and certified.
