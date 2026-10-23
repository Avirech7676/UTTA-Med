# Controlled Experiment Plan: SafeAdapt v2 Validation Matrix

**Document Version:** 2.0-Design  
**Date:** October 10, 2026  
**Auditor / Principal Investigator:** Independent Medical Imaging AI Auditor  
**Scope:** Controlled evaluation protocol comparing UTTA-Med v1 baselines against SafeAdapt v2 across ResNet-18 and ResNet-50 on Camelyon17-WILDS.  
**Strict Constraint:** *Planning document only. No GPU compute launched; no hypothetical numbers presented as measured results.*

---

## 1. Study Objectives & Research Hypotheses

1. **Primary Hypothesis ($H_1$):** SafeAdapt eliminates catastrophic collapse on worst-case seeds (Seeds 7 & 99 on ResNet-18; all seeds on ResNet-50), ensuring that target AUROC and $F_1$ never drop significantly below the frozen source model ($p_{\text{non-inferiority}} < 0.05$).
2. **Secondary Hypothesis ($H_2$):** A two-tier hybrid gate (fast deterministic confidence + conditional MC-Dropout arbitration) matches or exceeds v1 UTTA-Med adaptation quality while reducing inference latency by at least $80\%$.
3. **Tertiary Hypothesis ($H_3$):** Slide-aware episodic resetting prevents localized slide breakdown on tumor-dense WSIs (Slides 28 & 29) without sacrificing overall target metastasis detection.

---

## 2. Prespecified Primary Endpoint & Multiplicity Strategy

### 2.1 Primary Endpoint
The prespecified primary statistical endpoint is **Minimum-Seed Clinical Safety Non-Inferiority**:
$$\Delta_{\text{worst}} = \min_{s \in \mathcal{S}} [F_1^{\text{SafeAdapt}}(s) - F_1^{\text{Source}}(s)]$$
- **Success Criterion:** $\Delta_{\text{worst}} \ge -0.005$ (meaning on the worst random seed, SafeAdapt degrades $F_1$ by no more than $0.5\%$, compared to Tent which degrades $F_1$ by $-0.75$).

### 2.2 Secondary Co-Endpoints
1. **Mean $\Delta F_1$ across 5 seeds:** Superiority test against frozen Source model ($p < 0.05$).
2. **Mean Harm Rate:** Equivalence or reduction compared to v1 UTTA-Med ($\le 0.007 \pm 0.003$).
3. **Worst-Slide AUROC Degradation:** Maximum per-slide drop $\max_{w} (\text{AUROC}_{\text{source}}^w - \text{AUROC}_{\text{adapted}}^w) \le 0.05$.

### 2.3 Multiplicity Correction
- All pairwise hypothesis tests against the Source baseline and v1 UTTA-Med will be controlled using the **Holm–Bonferroni step-down procedure** at family-wise error rate $\alpha = 0.05$.
- Both seed-level paired $t$-tests ($df = 4$) and whole-slide clustered cluster-bootstrap confidence intervals ($B = 1,000$ resamples over $10$ WSIs) will be reported.

---

## 3. Experimental Matrix

### 3.1 Backbones & Seeds
- **Backbone Architectures:**
  1. `resnet18` (Primary benchmark, 1-logit BCE, dropout 0.5, 9.6k BN parameters).
  2. `resnet50` (Deep bottleneck benchmark, 1-logit BCE, dropout 0.5, 53.1k BN parameters).
- **Official Frozen Seeds:** $\mathcal{S} = \{42, 123, 2024, 7, 99\}$.

### 3.2 Evaluated Methods & Ablation Arms

| Method ID | Algorithm Name | Update Scope | Gating Strategy | Safety Safeguards | Compute Multiplier |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **M0** | **Source-Only (Frozen)** | None (0 parameters) | None | N/A (Baseline) | $1.0\times$ |
| **M1** | **Tent (Ungated)** | BN-affine ($\gamma, \beta$) | All patches ($100\%$ cov) | None | $1.0\times$ |
| **M2** | **EATA** | BN-affine | Entropy threshold | Fisher penalty (if precomputed) | $\sim 1.0\times$ |
| **M3** | **Confidence-Only Control** | BN-affine | Softmax $C(x) \ge \tau_C$ | None | $1.0\times$ |
| **M4** | **UTTA-Med v1 (Frozen)** | BN-affine | MC-Dropout $U(x) < \tau_U$ | Hard gate only | $20.0\times$ |
| **M5a** | **SafeAdapt (Core)** | BN-affine | Softmax $C(x) \ge \tau_C$ | Grad clip + $\ell_\infty$ parameter box | $1.0\times$ |
| **M5b** | **SafeAdapt (+ Alarm/Rollback)** | BN-affine | Softmax $C(x) \ge \tau_C$ | M5a + Running Prevalence Alarm + Rollback | $1.0\times$ |
| **M5c** | **SafeAdapt (+ Hybrid Gate)** | BN-affine | Two-tier (Fast $C$ + Arbiter $U$) | M5b + Conditional MC on borderline | $\sim 1.8\times$ |
| **M5d** | **SafeAdapt (+ Slide Reset)** | BN-affine | Two-tier | M5c + WSI Episodic Reset | $\sim 1.8\times$ |

---

## 4. Controlled Fair-Comparison Controls

### 4.1 Matched Adaptation Coverage Protocol
- Unconstrained methods (Tent) adapt on $100\%$ of patches, whereas selective methods adapt on a fraction $C \in [0.55, 0.70]$.
- To isolate the effect of the safety mechanism from the number of gradient steps:
  1. Measure the exact realized target coverage $C_{\text{SafeAdapt}}$ on each seed.
  2. Evaluate a **Random-gated control** forced to the exact same coverage $C_{\text{SafeAdapt}}$ on every run.
  3. Evaluate a **Confidence-only control** matched to the exact same percentile threshold.

### 4.2 Matched Computational Budget Protocol
- Wall-clock inference latency and total floating-point operations (FLOPs) must be logged per Whole-Slide Image on identical hardware (NVIDIA RTX / T4).
- Report:
  - Forward-pass inference time per 1,000 patches (ms).
  - Adaptation backward-pass time per batch (ms).
  - Total end-to-end WSI processing time (seconds per slide).

---

## 5. Comprehensive Metric Suite

Every experimental run must output a standardized JSON artifact containing:

1. **Discrimination:**
   - AUROC (Area under the Receiver Operating Characteristic curve).
   - AUPRC (Area under the Precision-Recall curve, sensitive to minority tumor patches).
2. **Clinical Decision Metrics (Operating at $\hat{y} = \mathbb{I}(p \ge 0.5)$):**
   - $F_1$ Score.
   - Sensitivity / Recall (identifying metastatic carcinoma).
   - Specificity (avoiding false alarms on normal lymph node tissue).
   - Precision (Positive Predictive Value).
3. **Clinical Safety & Reliability:**
   - Harm Rate: $\frac{\sum \mathbb{I}(\text{Source Correct} \land \text{Adapted Incorrect})}{\sum \mathbb{I}(\text{Source Correct})}$.
   - Correction Rate: $\frac{\sum \mathbb{I}(\text{Source Incorrect} \land \text{Adapted Correct})}{\sum \mathbb{I}(\text{Source Incorrect})}$.
   - Net Benefit / Flip Ratio: $\frac{\text{Corrections}}{\text{Harms}}$.
4. **Calibration & Uncertainty:**
   - Expected Calibration Error (ECE, 15 equal-width confidence bins).
   - Brier Score: $\frac{1}{N} \sum (p_i - y_i)^2$.
5. **Whole-Slide / Patient-Level Concordance:**
   - Slide-level tumor burden mean absolute error ($\text{MAE}_{\text{burden}} = \frac{1}{10} \sum |\bar{p}_s - y_{\text{true}, s}|$).
   - Worst-slide AUROC across the 10 hospital-2 WSIs.
6. **Execution Telemetry:**
   - Adaptation coverage ($C = N_{\text{adapted}} / N_{\text{total}}$).
   - Alarm trigger count (number of times rollback / fallback was engaged).
   - Maximum $\ell_\infty$ parameter displacement from source ($\max \|\theta - \theta_0\|_\infty$).

---

## 6. Development vs. Test Separation Protocol

```
┌─────────────────────────────────────────────────────────────┐
│  DEVELOPMENT PHASE (Hospitals 0, 3, 4 train; Hospital 1 OOD)│
│  - Tune: Gradient clip norm M_clip                          │
│  - Tune: Parameter bound ε_γ, ε_β                           │
│  - Tune: Alarm thresholds (π_min, Δz_max)                   │
│  - Tune: Hybrid gating cutoffs (τ_high, τ_low)              │
│  - Validate: Synthetic corruption stress test (Noise, Blur) │
│  - LOCK ALL HYPERPARAMETERS IN CONFIG YAML                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  FROZEN EVALUATION PHASE (Hospital 2 Test Set strictly)     │
│  - Execute single forward adaptation pass per seed          │
│  - Generate official metric JSONs                           │
│  - Run statistical tests & Holm-Bonferroni correction       │
│  - NO RETUNING PERMITTED                                    │
└─────────────────────────────────────────────────────────────┘
```
