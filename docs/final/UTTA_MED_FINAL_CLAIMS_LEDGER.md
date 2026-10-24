# UTTA-Med: Final Scientific Claims Ledger and Epistemic Calibration

**Audit Protocol:** Stage F.1 — Manuscript Integrity Freeze  
**Auditor Roles:** Senior Medical Imaging ML Researcher, Independent Scientific Auditor  
**Date:** October 2026  
**Standards:** STARD-AI, CLAIM, Good Machine Learning Practice (GMLP)

---

## 1. Epistemic Classification Schema

Every major finding, research claim, and architectural mechanism of the UTTA-Med project is evaluated under strict scientific boundaries and classified into one of four epistemic tiers:

- **Directly Supported:** Empirically confirmed by machine-readable data, valid statistical testing, or deterministic code invariants.
- **Partially Supported:** Demonstrated under restricted experimental settings, but subject to scope limitations or lacking generalizability across all domains.
- **Unsupported:** Lacks adequate empirical demonstration, underpowered statistically, or confounded by clustering/design constraints.
- **Contradicted by Results:** Empirically refuted by data collected during the project.

---

## 2. Definitive Claims Inventory and Epistemic Status

| # | Scientific Claim / Finding | Epistemic Classification | Precise Scientific Assessment |
|:---|:---|:---:|:---|
| **1** | **MC-Dropout Patch-Selection Redundancy** | **Partially Supported** | **MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting.** The Spearman rank correlation between max-softmax confidence and MC predictive variance was $\rho = -0.9917$ across evaluated Hospital-1 patches. However, this empirical redundancy in patch selection does not prove that MC-dropout cannot represent epistemic uncertainty in other regimes. Crucially, the pooled patch-level $p$-value ($p < 10^{-200}$) cannot be cited as independent inferential evidence because patches are clustered within five WSIs. |
| **2** | **Statistical Comparison of UTTA-Med vs Confidence-Only TTA** | **Directly Supported (Statistically Detectable, Practically Tiny)** | **The v1 Holm-adjusted $p=0.0430$ comparison indicates a small but statistically detectable paired F1 difference under the specified analysis.** Across 5 seeds, UTTA-Med achieved mean F1 of $0.830$ vs $0.831$ for Confidence-Only ($\Delta = -0.001013, 95\%\text{ CI: } [-0.0015, -0.0005]$). While Cohen's $d = -2.454$ reflects high consistency across seeds, the absolute difference of $\sim 0.001$ F1 is practically negligible. It does not constitute proof of meaningful clinical inferiority. |
| **3** | **Degradation Pattern on Slide 29** | **Partially Supported** | **Slide 29's performance degradation is consistent with confirmation bias.** Under severe domain shift on Slide 29, the model made high-confidence false positive predictions which were accepted by confidence gating ($\tau = 0.90$), accompanied by a drop in slide F1 from $0.3659 \to 0.3457$. However, our experiments do not establish a strict causal mechanism, which would require tracing individual misclassified updates into subsequent inference steps. |
| **4** | **Test-Time Adaptation on External Target Domain (Hospital 2)** | **Contradicted by Results (Demonstrates Negative Transfer)** | **Unsupervised confidence-gated adaptation degraded external target performance relative to the frozen source model.** On Hospital 2 (1,280 patches across 10 WSIs), Frozen Source achieved higher Macro Valid AUROC ($0.9308$ vs $0.9299$), Pooled AUROC ($0.9364$ vs $0.9360$), and Pooled F1 ($0.7840$ vs $0.7822$). Unsupervised TTA resulted in negative adaptation transfer across all evaluated metrics. |
| **5** | **SafeAdapt Bounded Parameter Updates and Numerical Handlers** | **Directly Supported (Within Evaluated Envelope)** | **SafeAdapt provides an explicitly bounded parameter-update mechanism with tested numerical checks and atomic rollback behavior.** Clamping at $\|\theta_t - \theta_0\|_\infty \le 0.05$ strictly bounded drift, with observed maximum displacement of $4.01 \times 10^{-5}$. Zero NaNs occurred in 50 streaming batches. However, zero NaNs in 50 batches and passing synthetic fault tests do not establish universal numerical stability across arbitrary data. |
| **6** | **Collapse Monitor Clinical Safety** | **Unsupported** | **Zero observed alarms across 40 evaluations on Hospital 2 establishes a low false-alarm rate under nominal streaming, NOT proof of clinical collapse detection.** Because benign adaptation with $\eta = 10^{-4}$ induced no true collapse, the detector had no opportunity to demonstrate clinical sensitivity. Detector sensitivity remains proven only under synthetic stress testing. |
| **7** | **Slide-Order Invariance via Boundary Reset** | **Directly Supported** | **WSI-boundary resets guarantee identical slide-level predictions regardless of slide ordering.** Re-running Hospital 1 under reversed and tumor-ascending permutations yielded bit-exact prediction identity, proving elimination of cross-slide state hysteresis. |
| **8** | **Clinical Deployment Readiness** | **Contradicted by Results** | **UTTA-Med and unsupervised TTA are not deployable clinically.** Unsupervised adaptation silently degrades diagnostic sensitivity on outlier patient slides, posing clinical risk without human oversight. |

---

## 3. Four Reconciled Overstatements (Audit Record)

### Correction A: Epistemic Uncertainty & MC-Dropout
- *Previous Draft:* "MC-Dropout Isolates Epistemic Uncertainty — Contradicted. Proven redundant with $p < 10^{-200}$."
- *Audited Correction:* "MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting."
- *Methodological Basis:* Rank correlation $\rho = -0.9917$ demonstrates selection redundancy in this specific ResNet-18 Camelyon17 benchmark, but does not provide a general theoretical refutation of MC dropout's capacity to estimate model uncertainty. Furthermore, patch-level inferential claims ($p < 10^{-200}$) are discarded due to spatial clustering within five WSIs.

### Correction B: Statistical Superiority vs Practical Significance
- *Previous Draft:* "UTTA-Med is significantly inferior to Confidence-Only TTA ($p = 0.0430, d = -2.454$)."
- *Audited Correction:* "A small but statistically detectable paired F1 difference ($\Delta \approx -0.001$ F1) under the specified analysis."
- *Methodological Basis:* A large effect size ($d = -2.454$) occurred because all 5 seeds showed a tiny, uniform decrease of between $0.0005$ and $0.0015$ F1. While statistically detectable under Holm-Bonferroni correction, an absolute difference of $0.001$ is clinically and practically negligible.

### Correction C: Confirmation Bias Causality
- *Previous Draft:* "Confidence Gating Fails to Prevent Confirmation Bias — Proven on Slide 29."
- *Audited Correction:* "Performance degradation on Slide 29 is consistent with confirmation bias."
- *Methodological Basis:* A strict causal proof would require an interventionist gradient-tracing experiment isolating the exact updates from misclassified patches and proving they induced the downstream errors. The observed drop ($0.3659 \to 0.3457$) is an associative finding consistent with the hypothesis.

### Correction D: Bounded Control vs Universal Numerical Safety
- *Previous Draft:* "SafeAdapt provides a mathematically verified clinical safety envelope."
- *Audited Correction:* "SafeAdapt provides an explicitly bounded parameter-update mechanism with tested numerical checks and atomic rollback behavior."
- *Methodological Basis:* Clamping parameters to an $L_\infty$ box and passing synthetic fault-injection tests provides an engineered containment envelope, but cannot guarantee universal numerical stability or clinical safety under unobserved data distributions.

---

## 4. Final Scientific Conclusion

The scientific integrity of the UTTA-Med project rests on transparent reporting:
1. Gating by MC-dropout variance was empirically redundant with softmax confidence for patch selection in this setting.
2. Unsupervised TTA failed to improve upon the frozen source model on the external target hospital.
3. Runtime safeguards successfully bound parameter drift, but cannot transform a negative adaptation trajectory into positive diagnostic generalization.
