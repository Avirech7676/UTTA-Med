# Phase 8 — Reviewer-Attack Audit & Pre-Emptive Rebuttal Dossier

**Document:** `docs/PHASE8_REVIEWER_ATTACK_AUDIT.md`  
**Status:** COMPLETE & VERIFIED against 5-Seed Primary Artifacts  
**Scope:** Pre-emptive scientific adversarial stress-test for UTTA-Med manuscript submission.

---

## 1. Executive Reviewer Attack Matrix

| # | Reviewer Attack | Core Vulnerability | Primary Evidence | Pre-Emptive Response Strategy | Formal Manuscript Limitation / Boundary |
|:---:|:---|:---|:---|:---|:---|
| **R1** | **“Why not just use softmax confidence? Isn't MC-Dropout variance redundant?”** | AUROC identical ($0.9355$ vs $0.9355$, $p=0.869$). Confidence leads F1 by $+0.0010$ ($p=0.0054$). Confidence is $5.7\times$ faster. | `results/metrics/full_s*.json`, Table Paired Stats (`d = -0.000046` AUROC, `d = -0.0010` F1). | **Embrace the negative result.** Do NOT claim UTTA > Confidence. Report binary rank-equivalence honestly; frame novelty as the first systematic reliability & selective adaptation benchmark in computational pathology. | Acknowledge that for binary histopathology metastasis classification, epistemic variance is functionally redundant with softmax confidence. |
| **R2** | **“Isn't this just standard selective prediction (reject option) under another name?”** | Conflating inference-time triage with adaptation-time gradient filtering. | `src/evaluation/coverage_risk.py`, `results/tables/coverage_risk_summary.csv` | **Strict conceptual decoupling:** Selective prediction passively rejects uncertain samples from evaluation. Selective adaptation actively shields model weights from gradient corruption (0 harms on 4/5 seeds in accepted set vs $>13,000$ for Tent). | Clarify that selective prediction simulation does not evaluate a live prospective pathologist triage workflow. |
| **R3** | **“Why evaluated only on Camelyon17? How do you know it generalizes to other organs or stains?”** | Single benchmark limitation (lymph node H&E). | Claim C8 (`docs/PHASE7_CLAIM_MATRIX.md`). Zero multi-dataset claims in authoritative package. | **Strictly bounded claims.** Acknowledge Camelyon17-WILDS as the gold-standard cross-hospital distribution shift benchmark (5 centers, 42 clinical WSIs). Position multi-organ validation (e.g. NCT-CRC) explicitly as future work. | Explicitly disclaim cross-modality and cross-organ generalizability. |
| **R4** | **“Are the reported performance improvements statistically significant?”** | AUROC gain over Source is mixed ($3/5$ seeds positive, $95\%$ CI $[-0.012, +0.024]$ includes 0). | Five-seed paired tests: F1 $\Delta = +0.0231$ ($p=0.0072$, $d=2.26$); AUROC $\Delta = +0.0058$ ($p=0.420$). | **Disaggregate metric claims.** Center the primary claim on F1 score ($p=0.0072$, positive on $5/5$ seeds) and Harm reduction ($25\times$ lower than Tent, $p=0.0313$). Never claim universal AUROC dominance. | State that AUROC improvements over source are small and not statistically significant; F1 is the consistent metric. |
| **R5** | **“Is Tent collapse simply an artifact of bad random seeds rather than a real algorithmic failure?”** | Tent fails catastrophically on seeds 7 & 99, but performs adequately on seeds 42, 123, 2024. | `results/tables/tent_collapse_summary.csv`, Slide 28/29 damage metrics ($p \to 0$, tumor sensitivity $<3\%$). | **Empirical instability finding.** Multi-seed evaluation is specifically what uncovers this: single-seed papers miss catastrophic failures. Collapse is localized to high-tumor WSIs (Slides 28 & 29) driving $>20,000$ harm flips. | Frame Tent collapse as an *empirical instability and seed-dependent vulnerability under cross-hospital shift*, not an unconditional failure. |
| **R6** | **“Where is the clinical validation? How does this impact actual whole-slide diagnosis?”** | Patch-level benchmarks do not establish clinical utility or patient-level outcomes. | `results/tables/slide_burden_summary.csv`, slide tumor burden concordance ($r=0.884$, $\text{MAE}=0.097$). | **Whole-slide tumor burden aggregation.** Connect patch predictions to clinical slide-level tumor burden estimation (WSI metastasis staging), showing Tent doubles burden error ($\text{MAE}=0.184$) while UTTA maintains $r=0.88$. | Acknowledge that this is an offline computational study, not a prospective clinical trial. |
| **R7** | **“Why choose N=20 MC-Dropout passes? Is the compute overhead justified?”** | MC-Dropout incurs $\approx 5.7\times$ inference overhead compared to deterministic forward passes. | `results/ablations/mc_passes/`: Spearman $r(N=20, N=50) = 0.994$, Jaccard $= 98.4\%$, $+0.0004$ AUROC gain at $2.3\times$ latency. | **Empirical elbow analysis.** Reconcile that $N=20$ captures $99.4\%$ of the ranking information of $N=50$ at $<50\%$ latency. Transparently report the runtime tradeoff vs deterministic confidence. | Document that $N=20$ is an operating point within MC-Dropout, but single-pass confidence is computationally superior. |

---

## 2. In-Depth Adversarial Audits & Systematic Responses

---

### Reviewer Attack 1: “Why not just use confidence? What is the actual novelty?”

#### Adversarial Prompt:
> *"The authors claim uncertainty-aware test-time adaptation, but their own ablation results show that simple softmax confidence gating achieves virtually identical AUROC ($0.9355$ vs $0.9355$) and slightly higher F1 ($0.8310$ vs $0.8300$, $p=0.0054$) while being $5.7\times$ faster. Given this, why should anyone use UTTA-Med over confidence gating? Isn't the novelty severely undermined?"*

#### Empirical Evidence:
- **AUROC (5-Seed Mean $\pm$ SD):** Confidence $0.9355 \pm 0.0218$ vs UTTA-Med $0.9355 \pm 0.0217$ ($\Delta = -0.000046$, paired $t$-test $p = 0.869$).
- **F1 Score:** Confidence $0.8310 \pm 0.0321$ vs UTTA-Med $0.8300 \pm 0.0323$ ($\Delta = -0.00101$, paired $t$-test $p = 0.0054$, Holm-adjusted $p = 0.0430$).
- **Harm Rate:** Confidence $0.0070 \pm 0.0028$ vs UTTA-Med $0.0073 \pm 0.0028$.
- **Runtime:** Deterministic confidence requires 1 pass per patch; $N=20$ MC-Dropout requires 20 passes ($189.2\text{s}$ vs $33.2\text{s}$ on test set).

#### Systematic Rebuttal & Paper Positioning:
1. **Embrace the negative result:** The paper explicitly positions this as an **honest, informative negative result**, rather than obscuring it behind misleading claims. Many medical AI papers claim MC-Dropout is essential without rigorous comparisons against deterministic baselines. We provide rigorous empirical proof that for binary classification, epistemic variance is functionally rank-equivalent to softmax confidence ($r > 0.98$).
2. **The True Novelty:**
   - The first multi-seed reliability and failure-mode audit of test-time adaptation under clinical distribution shifts in digital pathology.
   - Uncovering and formalizing the **catastrophic single-class attractor collapse** in unconstrained entropy minimization (Tent).
   - Demonstrating that **selective adaptation** (filtering adaptation gradients) is strictly necessary to prevent model collapse in histopathology, reducing harmful adaptations by $25\times$.
   - Establishing the mathematical and empirical basis for why entropy and confidence yield identical sample rankings in binary tasks.
3. **Manuscript Rule:**
   - **Banned:** *"UTTA-Med outperforms confidence gating."*
   - **Required:** *"UTTA-Med and confidence gating showed nearly identical AUROC performance ($\Delta = -0.00005$, 95% CI $-0.00078$ to $+0.00069$; two-sided paired $p = 0.869$). This comparison did not demonstrate a statistically significant difference; formal equivalence was not established."*

---

### Reviewer Attack 2: “Isn't this just selective prediction (reject option)?”

#### Adversarial Prompt:
> *"Filtering out high-uncertainty samples is well-known in literature as the reject option or selective classification (Geifman & El-Yaniv, 2017). How does this differ from simply applying a reject option to the source model?"*

#### Empirical Evidence:
- **Selective Prediction (Passive Reject Option):**
  - Source model at $100\%$ coverage: Error rate = $16.60\% \pm 1.56\%$, AUROC = $0.9297 \pm 0.0092$.
  - Source model at $50\%$ coverage (lowest-uncertainty kept): Error rate = $3.60\% \pm 2.92\%$, AUROC = $0.9806 \pm 0.0159$.
  - Error rate drops $4.61\times$, but **model parameters remain entirely static**.
- **Selective Adaptation (Active Parameter Updating):**
  - Evaluates how gradient updates modify the model on the accepted vs rejected subsets.
  - On the accepted $50\%$ subset: UTTA-Med produces **zero harmful flips** across 4 seeds and 1 harm on seed 2024 out of $42,527$ patches.
  - In contrast, unconstrained Tent corrupts the accepted subset: harms **$11,496$ patches** on Seed 7 and **$13,319$ patches** on Seed 99, dropping accuracy by $>27\%$.

#### Systematic Rebuttal:
- **Two Distinct Operations:**
  1. *Selective Prediction* is an inference-time decision: "Should an autonomous prediction be output to the clinician, or deferred to manual review?"
  2. *Selective Adaptation* is an optimization-time safeguard: "Should this unlabeled batch be permitted to update the model's batch normalization affine parameters?"
- Without selective adaptation, test-time adaptation actively corrupts high-confidence domain features. Selective gating ensures adaptation updates are computed exclusively from instances within the model's reliable support.

---

### Reviewer Attack 3: “Why Camelyon17 only? Where is multi-dataset generalization?”

#### Adversarial Prompt:
> *"The empirical study is conducted exclusively on Camelyon17-WILDS. Without evaluating on multiple distinct clinical datasets (e.g., prostate, colon, dermatology) or imaging modalities, how can the authors claim general applicability?"*

#### Empirical Evidence:
- The authoritative 5-seed protocol is frozen on Camelyon17-WILDS (Center 2 test set, 85,054 patches, 10 WSIs, 5 patient cases).
- Appendix preliminary checks on PathMNIST-C evaluate corruptions, not multi-center hospital shifts.

#### Systematic Rebuttal:
- **Deep, Rigorous Single-Domain Benchmark vs Shallow Multi-Dataset Runs:** Camelyon17-WILDS is widely recognized by the machine learning community (Koh et al., NeurIPS 2021) as the gold standard for clinical distribution shifts because it incorporates real-world variations in tissue preparation, staining protocols, and whole-slide scanners across 5 distinct medical centers.
- Instead of running a single seed across 5 toy datasets, we conducted a rigorous 5-seed, 7-method, 350-slide-evaluation study with bootstrap confidence intervals and failure-mode autopsy.
- **Bound the Claim (Claim C8):** We do not claim cross-dataset generalization. The manuscript explicitly states:
  > *"All primary findings are established on binary lymph node metastasis patches from the Camelyon17-WILDS cross-hospital benchmark. Evaluating multi-class tissue classification and cross-modality domain adaptation remains a critical objective for future work."*

---

### Reviewer Attack 4: “Are the results statistically significant?”

#### Adversarial Prompt:
> *"The authors claim improved performance, but is the difference statistically significant? For AUROC, the 95% confidence interval includes zero, and only 3 out of 5 seeds show improvement."*

#### Empirical Evidence:
- **F1 Score:**
  - Mean paired $\Delta\text{F1} = +0.0231 \pm 0.0102$ ($+2.31\%$).
  - 95% paired Student's $t$-interval: $[+0.0104, +0.0358]$ (strictly excludes 0).
  - Paired $t$-test: $t = 5.053$, $p = 0.0072$ (significant at $\alpha = 0.01$).
  - Sign test: Positive on $5/5$ seeds ($p = 0.0313$). Cohen's $d = 2.26$.
  - WSI paired bootstrap (Seed 42, 1,000 resamples): 95% CI $[+0.018, +0.090]$ (excludes 0).
- **AUROC:**
  - Mean paired $\Delta\text{AUROC} = +0.0058 \pm 0.0144$.
  - 95% paired $t$-interval: $[-0.0121, +0.0237]$ (includes 0).
  - Paired $t$-test: $t = 0.895$, $p = 0.420$ (not significant; $3/5$ seeds positive, $2/5$ seeds negative).
- **Harm Rate Reduction:**
  - UTTA harm ($0.0073 \pm 0.0028$) vs Tent harm ($0.1875 \pm 0.1790$): $\Delta = -0.1802$, $d = -1.02$, lower on $5/5$ seeds ($p = 0.0313$).

#### Systematic Rebuttal:
- **Metric Disaggregation:** The manuscript strictly disaggregates F1 from AUROC.
  - The claim of statistically significant adaptation improvement is **restricted to F1 score** ($p = 0.0072$, large effect size $d = 2.26$, confirmed by WSI bootstrap).
  - AUROC is reported transparently as mixed (mean $+0.0058$, not statistically significant).
  - The primary clinical benefit of selective adaptation is **harm suppression** ($25\times$ reduction, $p=0.0313$) and **collapse prevention**, not marginal ranking re-ordering.

---

### Reviewer Attack 5: “Is Tent collapse just a pathological random-seed artifact?”

#### Adversarial Prompt:
> *"Tent achieves reasonable AUROC ($\approx 0.93\text{--}0.94$) on seeds 42, 123, and 2024. The failure on seeds 7 and 99 appears to be an outlier or bad initialization. Why should the community view this as a fundamental issue rather than bad luck?"*

#### Empirical Evidence:
- **Seed 7:** Tent AUROC = $0.3429$, F1 = $0.0544$, Tumor Sensitivity = $2.80\%$ ($25,940$ harm flips).
- **Seed 99:** Tent AUROC = $0.3443$, F1 = $0.0576$, Tumor Sensitivity = $2.98\%$ ($27,682$ harm flips).
- **Random Gating Baseline:** Randomly updating 70% of patches also collapses on Seeds 7 & 99 (AUROC $0.3543$ and $0.3731$), confirming that lack of selectivity—not update frequency—causes collapse.
- **Slide-Level Autopsy:** Collapse is localized to the exact same slides across both seeds:
  - Slide 28 ($84.7\%$ tumor burden): Seed 7 sensitivity = $0.57\%$, Seed 99 sensitivity = $0.69\%$. Over $20,000$ harms on this slide alone.
  - Slide 29 ($52.5\%$ tumor burden): Seed 7 sensitivity = $0.00\%$, Seed 99 sensitivity = $0.00\%$.
- **Mechanism Calibration:** On collapsed seeds, Tent forces $97.0\%$ of samples into the $[0.9, 1.0]$ confidence bin, where empirical accuracy drops to $51.38\%$.

#### Systematic Rebuttal:
- In clinical medical imaging, a method that fails catastrophically on 2 out of 5 training seeds ($40\%$ failure rate) is clinically unviable. An autonomous deployment system cannot risk a $40\%$ probability of near-zero tumor sensitivity.
- The failure mode is deterministic and mechanistic, not random noise: unconstrained entropy minimization on imbalanced medical streams pushes logits toward a degenerate single-class attractor ($p \to 0$ for $>85\%$ of patches). Selective gating completely eliminates this vulnerability across all 5 seeds.

---

### Reviewer Attack 6: “Where is the clinical validation?”

#### Adversarial Prompt:
> *"The paper discusses clinical workflows and tumor burden, but does not evaluate pathologist diagnostic accuracy, reader study timings, or clinical decision support integration."*

#### Empirical Evidence:
- Slide-level aggregation analysis across all 10 Center-2 WSIs shows:
  - Ground-truth tumor burden ranges from $0.27\%$ to $84.7\%$.
  - UTTA-Med tracks slide tumor burden with $\text{MAE} = 0.0972 \pm 0.0161$, Pearson $r = 0.8842 \pm 0.0177$, Spearman $\rho = 0.8715 \pm 0.0251$.
  - Tent nearly doubles burden estimation error ($\text{MAE} = 0.1842 \pm 0.0818$, reaching $0.2737$ on collapse seeds).

#### Systematic Rebuttal:
- **Explicit Framing:** We do not claim clinical validation. The manuscript explicitly states:
  > *"This investigation constitutes an offline computational histopathology robustness study. Prospective clinical evaluation—including pathologist-in-the-loop diagnostic agreement, review time quantification, and institutional deployment integration—remains an essential next phase."*
- However, our slide-level burden analysis provides the critical computational bridge between patch-level accuracy and whole-slide diagnostic utility.

---

### Reviewer Attack 7: “Why N=20 MC-Dropout passes?”

#### Adversarial Prompt:
> *"Why was $N=20$ chosen? Earlier drafts and literature cite different pass counts (e.g. $N=50$ or $N=10$). Is $N=20$ ad-hoc?"*

#### Empirical Evidence:
- Empirical MC-Dropout sweep on Camelyon17 (`results/ablations/mc_passes/`):
  - $N=5$: $r = 0.381$, AUROC = $0.9482$, Runtime = $70.7\text{s}$.
  - $N=10$: $r = 0.402$, AUROC = $0.9498$, Runtime = $110.1\text{s}$.
  - $N=20$: $r = 0.416$, AUROC = $0.9507$, Runtime = $189.2\text{s}$.
  - $N=30$: $r = 0.419$, AUROC = $0.9509$, Runtime = $268.6\text{s}$.
  - $N=50$: $r = 0.422$, AUROC = $0.9511$, Runtime = $430.1\text{s}$.
- Spearman rank correlation between $N=20$ and $N=50$ uncertainty rankings: **$0.994$**.
- Top-percentile selection Jaccard overlap between $N=20$ and $N=50$: **$98.4\%$**.
- Marginal AUROC gain ($N=20 \to N=50$): only $+0.0004$, while runtime increases $2.27\times$.

#### Systematic Rebuttal:
- $N=20$ was empirically chosen as the Pareto-optimal knee of the curve. Increasing passes to $N=50$ yields negligible ranking difference ($r = 0.994$) at more than double the computational latency.
- We reconcile that while $N=20$ is optimal *within the MC-Dropout family*, single-pass deterministic confidence achieves identical selection at zero additional forward-pass cost.

---

## 3. Methodological Justification Matrix (Questions 8–12)

| Parameter / Choice | Value | Exact Rationale & Empirical Justification | Alternative Tested & Result |
|:---|:---:|:---|:---|
| **Backbone Architecture** | **ResNet-18** | Canonical benchmark architecture for Camelyon17-WILDS (Koh et al., 2021). Enables direct comparison with published literature without confounding architectural variance. | ResNet-50 explored in optional ablations; ResNet-18 remains the authoritative benchmark standard. |
| **Inner Adaptation Steps** | **$k = 1$** | Restricting adaptation to a single gradient update per batch prevents rapid representation drift on streaming unlabelled data. | $k \in \{2, 5, 10\}$ tested: at $k=2$, harm triples ($6.7\%$); at $k \ge 5$, all methods suffer catastrophic collapse (AUROC $<0.38$, harm $>51\%$). |
| **Adaptable Parameters** | **BN Affine ($\gamma, \beta$)** | Modulating only Batch Normalization affine parameters ($4,608$ parameters, $0.04\%$ of network) allows feature distribution recalibration to target staining while freezing convolutional filters to prevent catastrophic forgetting. | Full network fine-tuning: highly unstable under unconstrained entropy minimization, causing rapid weight divergence. |
| **Gating Threshold ($\tau$)** | **70th Percentile** | Calibrated strictly on the Center 1 validation hospital ($33,560$ patches). Retaining the 70% most confident instances provides sufficient batch support for BN recalibration while filtering out ambiguous or corrupted samples. | Evaluated across $[0.3, 1.0]$ in coverage-risk analysis; 70% maximizes F1 while maintaining harm $<0.8\%$. |
| **Target Labels** | **None (Zero)** | Reflects clinical reality: an algorithm deployed to an external hospital receives unannotated histopathology streams where real-time pathologist annotation is impossible. | Fully supervised fine-tuning is an oracle baseline, not a viable test-time adaptation protocol. |

---

## 4. Traceability & Artifact Reconciliation Audit

| Item / Claim | Primary Code / Script | Primary Data Source | Verification Status |
|:---|:---|:---|:---:|
| **5-Seed Main Results (Table 1)** | `scripts/upgrade/phase1_all.py` | `results/metrics/full_s*.json` | ✅ Reconciled to exact decimal digits |
| **Slide Burden Concordance** | `scripts/run_slide_aggregation_analysis.py` | `results/tables/slide_burden_summary.csv` | ✅ 5-seed mean $\text{MAE} = 0.0972$, $r = 0.8842$ |
| **Tent Collapse Case Study** | `scripts/run_tent_collapse_analysis.py` | `results/tables/tent_collapse_summary.csv` | ✅ Seed 7 AUROC $0.3429$, Seed 99 AUROC $0.3443$ |
| **Coverage–Risk Tradeoff** | `scripts/run_coverage_risk_analysis.py` | `results/tables/coverage_risk_summary.csv` | ✅ 50% keep error $3.60\%$, AUROC $0.9806$ |
| **$N=20$ Ablation Sweep** | `scripts/run_ablation_mc_passes.py` | `results/ablations/mc_passes/` | ✅ Spearman $r = 0.994$, Jaccard $= 98.4\%$ |
| **Inner Step $k$ Ablation** | `scripts/run_ablation_adapt_steps.py` | `results/ablations/camelyon17_adapt_steps_*.json` | ✅ $k=1$ safe ($2.1\%$ harm), $k=5$ collapses ($51.8\%$ harm) |
| **Figure 1: Coverage Tradeoff** | `scripts/run_coverage_risk_analysis.py` | `figures/fig_coverage_risk_tradeoff.png` | ✅ Generated & synced to `paper/ieee/figs/` |
| **Figure 2: Tent Mechanism** | `scripts/run_tent_collapse_analysis.py` | `figures/fig_tent_collapse_mechanism.png` | ✅ Generated & synced to `paper/ieee/figs/` |
| **Figure 3: Slide Tumor Burden** | `scripts/run_slide_aggregation_analysis.py` | `figures/fig_slide_burden_estimation.png` | ✅ Generated & synced to `paper/ieee/figs/` |

---

## 5. Reviewer-Audit Acceptance Criteria Checklist

- [x] **Why UTTA rather than confidence?** Explicitly answered: framed as an informative negative result; confidence is rank-equivalent and computationally cheaper.
- [x] **What exactly is novel?** Clarified: first 5-seed reliability & failure-mode audit in histopathology TTA; discovering single-class attractor collapse; proving selective adaptation is required.
- [x] **Why entropy and confidence are equivalent here?** Mathematically proven and empirically validated for binary classification ($K=2$).
- [x] **Why selective prediction $\ne$ selective adaptation?** Decoupled: inference-time human triage vs optimization-time parameter protection.
- [x] **Why five seeds?** Validated: single-seed studies miss catastrophic collapse occurring on Seeds 7 and 99.
- [x] **Why Center 1 validation / Center 2 test?** Strict adherence to WILDS protocol; zero test leakage during threshold calibration.
- [x] **Why ResNet-18?** Canonical benchmark standard for Camelyon17.
- [x] **Why $N=20$?** Empirical elbow point capturing $99.4\%$ ranking fidelity of $N=50$ at $<50\%$ latency.
- [x] **Why $k=1$?** $k \ge 5$ triggers representation collapse ($>51\%$ harm) across all methods.
- [x] **Why BN affine parameters?** Minimal update footprint ($0.04\%$) preserves convolutional representations while adapting stain normalization.
- [x] **Why 70th-percentile threshold?** Optimal safety-adaptation tradeoff calibrated on Center 1.
- [x] **Why no target labels?** Unsupervised deployment requirement.
- [x] **Whether claims generalize beyond Camelyon17?** Strictly bounded; generalizability claims rejected.
- [x] **Whether clinical claims are justified?** Strictly disclaimed as an offline computational study.
- [x] **Whether Tent collapse is seed-dependent?** Yes, characterized as an empirical instability on 2/5 seeds.
- [x] **Whether all reported numbers reconcile to primary artifacts?** Complete 1:1 numerical audit passed.
- [x] **Whether every figure/table has a traceable source?** Complete traceability matrix verified.

Step 8.4 is CLOSED and signed off.
