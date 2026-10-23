# Go / No-Go Criteria & Decision Framework: SafeAdapt v2

**Document Version:** 2.0-Governance  
**Date:** October 10, 2026  
**Auditor / Governance Authority:** Independent Medical Imaging AI Auditor & Scientific Lead  
**Scope:** Formal milestone gates governing whether to advance SafeAdapt from Phase 1 (forensics & design) to Phase 2 (code implementation) and Phase 3 (frozen benchmark evaluation).

---

## 1. Governance Principles

Before any GPU computation or evaluation is initiated, all quantitative decision gates must be pre-registered. SafeAdapt will be judged by **clinical safety and statistical non-inferiority**, not by whether it claims an inflated, unverified accuracy gain.

---

## 2. Formal Milestone Decision Gates

### Gate 1: Development Feasibility & Safety Checks (Hospital 1 OOD Validation)
*Evaluated exclusively on Hospital 1 without touching Hospital 2.*

| Criterion ID | Evaluation Metric | Threshold for "GO" | Threshold for "NO-GO" | Verification Evidence |
| :---: | :--- | :--- | :--- | :--- |
| **G1.1** | **Numerical Stability (ResNet-50)** | **Zero NaNs** across all output probabilities, loss values, and metric evaluations on Hospital 1. | Any `NaN` or `inf` during forward or backward pass. | Automated assertion in test pipeline. |
| **G1.2** | **Alarm Sensitivity on Synthetic Failure** | Collapse alarm triggers within $\le 5$ batches when an adversarial drift or severe corruption (Gaussian noise 3) is injected. | Alarm fails to detect negative logit collapse ($p \to 0$). | Synthetic drift unit tests. |
| **G1.3** | **Zero False Alarms on Clean Data** | Alarm trigger rate $< 0.1\%$ on clean Hospital 1 validation stream. | False alarm rate $> 2.0\%$ (unnecessary rollbacks). | Hospital 1 validation log audit. |
| **G1.4** | **Computational Efficiency** | Hybrid fast-gate achieves $\ge 5\times$ speedup over full MC-Dropout ($N=20$) while maintaining $\ge 98\%$ gating concordance. | Speedup $< 3\times$ or concordance $< 95\%$. | Wall-clock latency benchmark. |

**Gate 1 Decision Rule:** All four criteria (G1.1–G1.4) must achieve **PASS** status. A single failure mandates architectural redesign before touching target test data.

---

### Gate 2: Target Evaluation Non-Inferiority (Hospital 2 Test Set)
*Evaluated on the frozen 5-seed benchmark after Gate 1 passes.*

| Criterion ID | Evaluation Metric | Threshold for "GO" (Success) | Threshold for "NO-GO" (Failure) |
| :---: | :--- | :--- | :--- |
| **G2.1** | **Worst-Seed $F_1$ Clinical Non-Inferiority** | $\min_{s} (F_1^{\text{SafeAdapt}} - F_1^{\text{Source}}) \ge -0.005$ across all 5 seeds on ResNet-18 and ResNet-50. | Any seed exhibits $F_1$ drop $> 0.010$ (e.g. Tent's drop of $-0.75$). |
| **G2.2** | **Elimination of Single-Class Collapse** | Zero seeds exhibit $\text{Recall} < 0.20$ or $\text{Specificity} > 0.999$ with $F_1 < 0.10$. | Single-class negative attractor observed on any seed. |
| **G2.3** | **Mean Harm Rate Suppression** | Mean harm rate across 5 seeds $\le 0.008$ ($< 0.8\%$ flip rate). | Mean harm rate $> 0.020$ ($> 2.0\%$). |
| **G2.4** | **Superiority over Ungated Tent on Collapse Seeds** | On Seeds 7 and 99, SafeAdapt AUROC $> 0.900$ (Tent is $\approx 0.343$). | SafeAdapt AUROC falls below $0.850$ on Seeds 7 or 99. |
| **G2.5** | **Whole-Slide Tumor Burden Calibration** | Whole-slide mean absolute error $\text{MAE}_{\text{burden}} \le 0.110$ across all 10 test WSIs. | $\text{MAE}_{\text{burden}} > 0.150$ (Tent reaches $0.184$). |

---

## 3. Strict Red Lines (Immediate Abort Conditions)

The v2 SafeAdapt extension must be immediately aborted and declared unviable if any of the following occur:

1. **Target-Label Contamination:** Any hyperparameter, threshold, or decision rule is tuned using Hospital 2 labels.
2. **Failure to Beat Frozen Source on Worst-Case Scenarios:** If SafeAdapt causes a worse performance drop than simply doing nothing (keeping the pre-trained source model frozen), the safety invariant is violated.
3. **Loss of v1 Artifact Integrity:** Any modification, overwriting, or silent alteration of frozen v1 checkpoints, tables, raw JSONs, or manuscript claims.
4. **ResNet-50 Persistent Degeneracy:** If ResNet-50 cannot achieve $F_1 > 0.70$ and $\text{AUROC} > 0.90$ under SafeAdapt on Hospital 2, ResNet-50 must be permanently quarantined as an unsuitable architecture for naive TTA under this acquisition shift.

---

## 4. Governance Signoff Checklist

- [x] All 5 criteria in Gate 1 defined with pre-registered mathematical formulas.
- [x] All 5 criteria in Gate 2 defined with explicit numerical tolerances.
- [x] Explicit separation between development (Hospital 1) and test (Hospital 2) enforced.
- [x] No v1 results altered or invalidated.
- [x] Reviewer attack risks pre-empted by honest negative-result framing.
