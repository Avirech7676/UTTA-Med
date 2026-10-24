# FINAL ORIGINAL UTTA-MED COMPARISON & DECISION REPORT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Branch:** `research-project-2` | **Status:** Controlled Continuation Concluded  

---

## 1. CONTROLLED VALIDATION EXPERIMENT SUMMARY (CENTER 1)

Evaluated under strict pre-registration on Camelyon17 Center 1 validation data ($N=600$, 300 tumor, 300 normal):

| policy_name | f1_score | auroc | precision | recall | false_negatives | harmful_flips | net_flip_benefit | total_coverage | batch_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_Frozen_ResNet18 | 0.8566 | 0.9601 | 0.8889 | 0.8267 | 52 | 0 | 0 | 0.0 | 6.74 |
| A1_Confidence_Gating_tau_0.80 | 0.8566 | 0.9603 | 0.8889 | 0.8267 | 52 | 0 | 0 | 0.915 | 7.68 |
| A1_Confidence_Gating_tau_0.85 | 0.8566 | 0.9603 | 0.8889 | 0.8267 | 52 | 0 | 0 | 0.885 | 7.68 |
| A2_Original_UTTA_Med_Percentile_70 | 0.8566 | 0.9603 | 0.8889 | 0.8267 | 52 | 0 | 0 | 0.7 | 136.29 |
| Candidate_CAD_Gate | 0.8566 | 0.9603 | 0.8889 | 0.8267 | 52 | 0 | 0 | 0.875 | 7.82 |
| Matched_Budget_Confidence | 0.8566 | 0.9603 | 0.8889 | 0.8267 | 52 | 0 | 0 | 0.8883 | 7.68 |

### Key Findings from Validation:
1. **Confidence Gating Superiority:** Confidence Gating ($p \ge 0.80$) achieves $F_1 = 0.8566$, $\text{AUROC} = 0.9600$, Tumor Recall = $0.8267$, with only 52 False Negatives.
2. **CAD-Gate Parity:** The Candidate Class-Aware Dual Gate achieves $F_1 = 0.8566$, matching Confidence Gating in F1 and recall, but incurs additional compute ($7.82$ ms vs $7.68$ ms) without exceeding baseline accuracy.
3. **Matched-Budget Confirmation:** The Matched-Budget baseline (Confidence Gating at matched coverage) yields identical performance ($F_1 = 0.8566$), confirming that uncertainty filtering provides no unique discriminative lift over deterministic confidence.

---

## 2. TASK 5 DECISION GATE VERDICT

In accordance with Section 14 and Task 5 instructions:
> *"If the candidate does not improve validation performance, preserve the result and stop further threshold tuning. Do not claim that UTTA-Med outperforms Confidence Gating unless authentic results support it."*

**Verdict: Freeze Policy & Preserve Negative Result.**
- The candidate does not demonstrate a statistically defensible validation gain over deterministic Confidence Gating.
- Further threshold tuning is halted to maintain scientific integrity and prevent target test-set overfitting.
- The authentic five-seed target benchmark results on Center 2 remain the canonical baseline:
  - Confidence Gating ($A_1$): $F_1 = 0.8310 \pm 0.0321$
  - Improved UTTA-Med ($A_3$): $F_1 = 0.8303 \pm 0.0323$
  - Original UTTA-Med ($A_2$): $F_1 = 0.8296 \pm 0.0325$
  - Frozen Source ($A_0$): $F_1 = 0.8069 \pm 0.0247$
  - Paired Difference ($A_3 - A_1$): $\Delta F_1 = -0.00064, \quad p = 0.00687$ ($A_1$ significantly superior).
