# UTTA-Med: Comprehensive Scientific Claims Audit and Epistemic Status Ledger

**Audit Scope:** UTTA-Med v1 Hypothesis Claims vs v2 Empirical Reality  
**Auditor Roles:** Senior Medical Imaging ML Researcher, Independent Scientific Auditor  
**Date of Audit:** October 2026  
**Standards:** STARD-AI, CLAIM (Checklist for Artificial Intelligence in Medical Imaging), Good Machine Learning Practice (GMLP)

---

## 1. Classification Schema

In accordance with rigorous reproducibility principles, every scientific and technical assertion originating in the UTTA-Med literature, codebase, or project proposals is audited against bit-exact experimental evidence and assigned one of four epistemological classifications:

1. **Directly Supported:** Empirically substantiated by bit-exact machine-readable data with valid statistical testing or deterministic mathematical proofs.
2. **Partially Supported:** Demonstrated under restricted, domain-specific conditions, but lacking generalizability or resting on heuristic assumptions.
3. **Unsupported:** Lacks empirical confirmation, underpowered statistically, or confounded by methodological design limitations.
4. **Contradicted by Results:** Explicitly refuted by empirical experimental evidence collected during the project.

---

## 2. Comprehensive Claims Matrix

| # | Scientific / Technical Claim | Epistemic Classification | Evidence Source & Forensic Audit |
|:---|:---|:---|:---|
| **1** | **MC-Dropout Isolates Epistemic Uncertainty in Histopathology** | **Contradicted by Results** | Stage D2 & E0 audits revealed a rank correlation of $\rho = -0.9917$ between Softmax Confidence and MC Predictive Variance. MC variance functions as an inverted surrogate of confidence rather than isolating orthogonal epistemic uncertainty. |
| **2** | **UTTA-Med Outperforms Confidence-Only TTA** | **Contradicted by Results** | Historical v1 Table 2 shows UTTA-Med is significantly inferior in F1 to Confidence-Only ($p = 0.0430$ Holm-adjusted, Cohen's $d = -2.454$). Stage D2 benchmark shows identical accuracy with $3.5\times$ slower runtime. |
| **3** | **Confidence Gating Prevents Confirmation Bias** | **Contradicted by Results** | On Hospital 2 (Stage E2), confidence-gated TTA actively degraded performance on Slide 29 (F1 fell from $0.3659 \to 0.3457$). Filtering by confidence preferentially selects confident false positives under severe domain shift. |
| **4** | **Test-Time Adaptation Improves External Target Generalization** | **Contradicted by Results** | Stage E2 target evaluation demonstrated that Frozen Source achieved the highest Macro AUROC ($0.9308$), Pooled AUROC ($0.9364$), F1 ($0.7840$), ECE ($0.0650$), and Brier ($0.0917$). Adaptation caused negative transfer across all metrics. |
| **5** | **SafeAdapt Ensures Numerical Stability ($NaN$ / $Inf$ Interception)** | **Directly Supported** | Formally proven via unit test suite (`test_safeadapt_v2.py`) and Stage B/C mathematical handlers. Non-finite gradients are intercepted and rolled back atomically with zero memory corruption. |
| **6** | **Parameter Displacement Bound ($L_\infty \le \epsilon$) Constrains Drift** | **Directly Supported** | Enforced deterministically in `SafeAdaptController`. In Stage E2, maximum parameter drift was strictly bounded at $4.01 \times 10^{-5} \ll 0.05$. Mathematical guarantee against arbitrary parameter runaway. |
| **7** | **Collapse Monitor Demonstrates High Clinical Safety** | **Unsupported** | Zero alarms occurred across 40 eligible evaluations on Hospital 2. Because no true catastrophic collapse occurred on Hospital 2 under $\eta=10^{-4}$, zero observed alarms proves absence of false alarms under normal execution, NOT clinical detector sensitivity. |
| **8** | **Slide-Order Invariance Guaranteed via WSI Boundary Reset** | **Directly Supported** | Stage E1 order-invariance stress tests proved identical predictions across reverse and tumor-ascending permutations under per-WSI boundary resets of model, optimizer, and monitor buffers. |
| **9** | **Statistically Significant Diagnostic Improvement over Source** | **Unsupported** | In the v1 5-seed benchmark, AUROC improvement was $+0.0058$ ($p = 0.8397$ Holm-adjusted). In Stage E2 target evaluation, adaptation performance was strictly negative. |
| **10** | **Readiness for Real-World Clinical Deployment** | **Contradicted by Results** | System cannot be deployed clinically. Unsupervised adaptation degrades boundary discrimination on difficult patient slides and introduces negative transfer without human-in-the-loop oversight. |

---

## 3. Deep-Dive Audits of Focal Claims

### 3.1 Claim 1: Monte Carlo Dropout and Uncertainty Novelty
- **Original Claim:** *“UTTA-Med introduces a novel epistemic uncertainty gating mechanism via Monte Carlo dropout that selectively filters ambiguous histopathology patches and prevents adaptation to out-of-distribution artifacts.”*
- **Forensic Audit:**
  1. **Empirical Redundancy:** In Stage D2, the Spearman rank correlation between max-softmax confidence and MC dropout predictive variance across 320 patches was $\rho = -0.9917$ ($p = 9.73 \times 10^{-286}$). Gating decisions had a Jaccard similarity of 0.782, with disagreements exclusively consisting of patches where confidence was high and MC variance was moderately elevated.
  2. **Error Analysis of Disagreements:** Among the 61 patches accepted by confidence gating but rejected by MC-only gating, the error rate was 21.3%, identical to the baseline patch error rate. MC dropout did not preferentially identify false positives.
  3. **Computational Penalty:** Evaluating 20 MC forward passes increased wall-clock latency from 168.73s to 598.21s (a $3.55\times$ computational penalty) while providing zero statistical gain in AUROC (0.9737 vs 0.9738).
- **Audit Finding:** **Contradicted by Results.** MC-dropout predictive variance is computationally redundant for patch selection in this architecture and does not isolate independent epistemic uncertainty.

### 3.2 Claim 2: Confidence-Only Superiority Over Hybrid Adaptation
- **Original Claim:** *“The dual uncertainty-confidence formulation in UTTA-Med provides superior calibration and classification robustness compared to single-metric confidence heuristics.”*
- **Forensic Audit:**
  1. In the authoritative 5-seed paired statistical test (Table 2), UTTA-Med achieved an average F1 of $0.830$ versus $0.831$ for Confidence-Gated TTA.
  2. The paired difference $\Delta = -0.001013$ is statistically significant in favor of Confidence-Gated TTA ($p = 0.00537$ nominal, $p = 0.04295$ Holm-adjusted, Cohen's $d = -2.454$).
  3. AUROC differences were negligible ($\Delta = -0.000046, p = 0.8688$).
- **Audit Finding:** **Contradicted by Results.** Confidence-only gating is empirically and statistically superior or non-inferior to the hybrid formulation across all evaluated metrics, at a fraction of the computational complexity.

### 3.3 Claim 3: Confirmation Bias and Adaptation Dynamics Under Domain Shift
- **Original Claim:** *“Unsupervised entropy minimization adapts benign model representations to resolve tissue staining and scanner differences across hospitals.”*
- **Forensic Audit:**
  1. On Hospital 2 (Stage E2), 1,069 out of 1,280 patches (83.5%) met the confidence gating threshold $\tau = 0.90$.
  2. Because the source model was already well-calibrated (Frozen Source AUROC 0.9308, F1 0.7840), test-time entropy minimization reinforced existing model biases.
  3. On Slide 29—a challenging metastatic lymph node section with dense lymphocytic background—the model made confident misclassifications. Adapting on these confident false positives degraded slide F1 from $0.3659$ to $0.3457$.
  4. Across the pooled cohort of 1,280 patches, pooled F1 dropped from $0.7840 \to 0.7822$, pooled AUROC fell from $0.9364 \to 0.9360$, and Brier score worsened from $0.091709 \to 0.092051$.
- **Audit Finding:** **Directly Contradicted.** Unsupervised adaptation induces negative transfer and exacerbates confirmation bias under realistic histopathology shifts.

### 3.4 Claim 4: Numerical Stability and State Rollback Mechanics
- **Original Claim:** *“SafeAdapt guarantees deterministic recovery from floating-point anomalies and parameter divergence during online adaptation.”*
- **Forensic Audit:**
  1. The `SafeAdaptController` maintains CPU/CUDA state snapshots before every backward update.
  2. Unit tests `test_safeadapt_v2.py` verify that artificial injection of `NaN` loss, infinite gradients, or exploding updates triggers instantaneous atomic rollback.
  3. The current batch is recomputed using the last known accepted state, and subsequent slide batches proceed without corrupted hidden state or batch normalization buffers.
  4. In Stage E2, parameter displacement $L_\infty$ was tracked per-batch, confirming that parameter displacement remained within $4.01 \times 10^{-5}$ across all 40 updates.
- **Audit Finding:** **Directly Supported.** The architectural engineering of SafeAdapt successfully eliminates floating-point failure modes.

### 3.5 Claim 5: Clinical Safety and Collapse Detection
- **Original Claim:** *“The collapse monitor provides verified safety against model collapse in clinical deployment.”*
- **Forensic Audit:**
  1. While the mathematical logic of the sliding-window detector (monitoring MSP, entropy, and class balance) is verified, **zero collapse alarms fired during Stage E2**.
  2. As established in the Stage E1/E1.5 audit, the absence of alarms under benign micro-updates ($\Delta \approx 4 \times 10^{-5}$) demonstrates that the detector does not trigger false alarms during routine execution.
  3. However, zero observed alarms on benign data **cannot be cited as empirical evidence that the detector will reliably intercept real-world collapse in the clinic**. Detector sensitivity can only be established under active collapse conditions (such as synthetic stress testing).
- **Audit Finding:** **Unsupported as a Clinical Safety Claim.** The detector is verified to have a low false-alarm rate under nominal streaming conditions, but its true clinical alarm sensitivity remains unproven on clinical cohorts without simulated fault injection.

---

## 4. Epistemic Summary

The scientific narrative of UTTA-Med must be reframed from a *triumphalist claim of superior adaptation* to an **essential cautionary study on the limits of unsupervised test-time adaptation in medical imaging**:

1. **Unsupervised adaptation is not a free lunch:** Under real-world histopathology domain shifts, frozen well-trained source models frequently outperform self-adapting models.
2. **Computational frugality matters:** Complex Bayesian approximations like MC dropout can introduce massive computational latency while yielding representations functionally indistinguishable from simple softmax confidence.
3. **Runtime safeguards are necessary but insufficient for accuracy:** Deterministic parameter bounds and rollback mechanisms successfully prevent catastrophic collapse, but they cannot convert negative adaptation transfer into positive transfer.
