# Stage D2: Scientific Evaluation and Go / No-Go Decision

**Author:** Senior Medical Imaging ML Researcher, PyTorch Performance Engineer, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Official Scientific Audit Determination  
**Target:** SafeAdapt Gating Architecture (UTTA-Med v2)  

---

## 1. Executive Summary and Decision

### Official Recommendation: **NO-GO for MC-Dropout / Hybrid Gating; PIVOT TO CALIBRATED CONFIDENCE GATING**

In accordance with the prespecified decision criteria:
> *"NO-GO for the hybrid research direction if confidence-only performs comparably with lower cost and MC provides no demonstrated incremental value. Do not manufacture a positive result. A negative finding is scientifically valid."*

Based on comprehensive evaluation across Experiments A, B, C, and D on Hospital 1:
1. **MC-Dropout Predictive Variance Adds Zero Selection Value:**
   - Spearman rank correlation with deterministic confidence is $\rho = \mathbf{-0.9917}$ ($p = 9.73 \times 10^{-286}$).
   - In selection mask analysis, MC-Only selected **zero (0)** candidate patches that Confidence-Only rejected.
2. **Confidence-Only Outperforms on the Efficiency Frontier at Matched Budget:**
   - Under an approximately matched budget ($68.8\%$ vs. $68.1\%$ coverage, exactly $10/10$ accepted optimizer steps), Confidence-Only achieved **identical predictive performance** to MC-Only:
     - **AUROC:** $0.9737$ vs. $0.9737$
     - **AUPRC:** $0.9544$ vs. $0.9544$
     - **F1 Score:** $0.8632$ vs. $0.8632$
   - Confidence-Only executed in **118.09 seconds**, compared to **518.92 seconds** for MC-Only—delivering a **$4.39\times$ wall-clock speedup** with zero stochastic passes.
3. **Hybrid Gating Solves the Wrong Problem:**
   - Hybrid gating successfully reduced MC patch-passes by $86.6\%$ and yielded a $3.10\times$ speedup over MC-Only.
   - However, because MC-dropout predictive variance itself provides no discriminative advantage over deterministic confidence, optimizing MC sampling with a two-stage hybrid policy adds algorithmic and hyperparameter complexity without clinical or statistical justification.

---

## 2. Synthesis of Experimental Findings

### Experiment A — Natural Operating Policies
| Policy | Patch Coverage | Accepted Steps | AUROC | F1 | ECE | Latency (s) | Speedup vs MC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Source** | 0.0% | 0 | 0.9736 | 0.8632 | 0.0786 | 72.09s | — |
| **Confidence-Only ($\tau=0.90$)** | 87.5% | 10 | 0.9738 | 0.8596 | 0.0789 | 204.88s | $3.05\times$ |
| **MC-Only ($K=20$)** | 68.1% | 10 | 0.9737 | 0.8596 | 0.0846 | 624.86s | $1.00\times$ (Ref) |
| **Hybrid ($0.65/0.95$)** | 83.8% | 10 | 0.9738 | 0.8632 | 0.0788 | 201.60s | **$3.10\times$** |

### Experiment B — Matched Adaptation Budget
| Policy | Target Budget Match | Coverage (%) | Accepted Steps | AUROC | F1 | Latency (s) | Speedup vs MC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Confidence-Only ($\tau=0.9931$)** | Residual: +0.6% (+2) | 68.8% | 10 / 10 | **0.9737** | **0.8632** | **118.09s** | **$4.39\times$** |
| **MC-Only ($K=20$)** | Reference Target | 68.1% | 10 / 10 | **0.9737** | **0.8632** | 518.92s | $1.00\times$ |
| **Hybrid ($0.60/0.9969$)** | Residual: -0.9% (-3) | 67.2% | 10 / 10 | **0.9737** | **0.8632** | 432.92s | $1.20\times$ |

### Experiment C — Selection Complementarity
- **Spearman $\rho$:** $-0.9917$
- **Jaccard Similarity:** $0.7821$ (Conf vs MC), $0.9536$ (Conf vs Hybrid)
- **MC-Only Exclusive Candidates:** **0 patches**
- **Offline Error Rate on Conf-Only Exclusive (61 patches):** $21.31\%$
- **Offline Error Rate on Hybrid Overrides (48 patches):** $20.83\%$

### Experiment D — Computational Cost
- **MC-Only:** 6,400 stochastic forward patch-passes; 624.86s total elapsed time.
- **Natural Hybrid:** 860 stochastic forward patch-passes; 201.60s total elapsed time ($3.10\times$ speedup).
- **Matched Confidence:** **0 stochastic forward patch-passes; 118.09s total elapsed time ($4.39\times$ speedup)**.

---

## 3. Scientific Defense of the Negative Result

In clinical machine learning, negative results that dismantle unnecessary complexity are of equal or greater value than positive claims of marginal speedups:
1. **Scientific Honesty Over Feature Creep:**
   MC-dropout was originally hypothesized to capture epistemic uncertainty and prevent adaptation on out-of-distribution tissue. On Camelyon17 ResNet-18, the stochastic dropout variance is almost perfectly colinear ($\rho = -0.9917$) with softmax probability confidence.
2. **Computational Sustainability in Clinical Pathology:**
   Running 20 forward passes per tile across whole slide images containing thousands of patches is computationally prohibitive. Hybrid gating was designed to mitigate this, but since Calibrated Confidence-Only achieves the exact same protection with zero stochastic forward passes, retaining MC-dropout in any form is scientifically indefensible.
3. **Provenance and Non-Manufacturing:**
   We strictly refuse to manufacture a rationale for Hybrid gating. The data demonstrates that Calibrated Confidence Gating ($\tau \approx 0.99$) is strictly superior on Pareto efficiency.

---

## 4. Prescribed Next Steps for UTTA-Med v2

1. **Retire MC-Dropout and Hybrid Architectures:**
   Do not promote MC-Only or Hybrid gating to Hospital-2 evaluation. Freeze their implementations in `src/v2/gating/` as historical benchmark artifacts.
2. **Adopt Calibrated Confidence Gating with SafeAdapt Controller:**
   The production candidate for Stage E (Hospital-2 Target Validation) shall be:
   - **Controller:** Stage B Atomic SafeAdapt Controller with Stage C1 Collapse Monitor.
   - **Gating Policy:** Calibrated Confidence-Gating ($\tau_{\text{conf}} \approx 0.99$).
   - **Guarantees:** Frozen source preservation, atomic parameter rollback on non-finite gradients, zero stochastic pass overhead.
3. **Prepare Stage E Protocol:**
   Proceed to preregister the Hospital-2 target evaluation protocol under the calibrated confidence-gated SafeAdapt architecture.
