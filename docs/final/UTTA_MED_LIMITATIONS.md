# UTTA-Med: Comprehensive Limitations, Protocol Shifts, and Clinical Boundaries

**Audit Context:** UTTA-Med v2 — Stage F Final Scientific Consolidation  
**Auditor Roles:** Senior Medical Imaging ML Researcher, Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Mandatory Disclosure and Technical Transparency Statement

---

## 1. Overview and Purpose

Scientific integrity in medical imaging machine learning requires an exhaustive, unvarnished disclosure of methodological boundaries, statistical constraints, protocol evolutions, and clinical contraindications. This document catalogs all known limitations of the UTTA-Med project, categorizing them into:
1. Protocol shifts between v1 and v2.
2. Sampling budget and denominator constraints.
3. Target domain exposure and statistical power limitations.
4. Algorithmic and architectural failure modes.
5. Absolute clinical and ethical deployment contraindications.

---

## 2. Protocol Shifts: v1 vs v2

A frequent source of irreproducibility in machine learning literature is unannounced protocol drift between development iterations. Table 1 explicitly contrasts the operational semantics of UTTA-Med v1 and v2:

| Dimension | Historical UTTA-Med v1 | Refactored SafeAdapt v2 | Scientific Implication |
|:---|:---|:---|:---|
| **Prediction Timing** | **Post-Adaptation** Evaluation (Batch evaluated after adapting on it) | **Pre-Adaptation** Streaming Evaluation (Batch evaluated before updating on it) | Post-adaptation evaluation allows a batch to benefit from immediate self-adaptation, masking boundary corruption. Pre-adaptation evaluation faithfully simulates real-world online inference. |
| **Parameter Constraints** | Unconstrained optimization (Loss thresholding only) | Strict $L_\infty \le 0.05$ boundary relative to source parameters $\theta_0$ | Prevents parameter explosion and runaway drift regardless of loss magnitude. |
| **Numerical Safeguards** | None (Vulnerable to $NaN$/$Inf$ crashing) | Non-finite gradient detection, atomic state rollback | Guarantees system resilience against floating-point corruption. |
| **Failure Recovery** | None (Model remained degraded) | Rolling collapse monitor with two-stage escalation & source fallback | Deterministic containment envelope against catastrophic collapse. |
| **Boundary Resets** | Continuous streaming or ad-hoc resets | Strict per-WSI resets of model, optimizer, and monitor | Eliminates cross-slide contamination and batch-order hysteresis. |
| **Uncertainty Engine** | Monte Carlo dropout (20 forward passes) | Max-softmax confidence thresholding ($\tau = 0.90$) | Removes $3.5\times$ computational overhead after proving MC variance redundancy ($\rho = -0.992$). |

---

## 3. Sampling Budget and Denominator Limitations

### 3.1 Subsampling vs Full Whole-Slide Evaluation
In routine clinical pathology practice, a single whole-slide lymph node specimen digitized at $40\times$ magnification yields between **20,000 and 60,000 tissue patches** (each $224 \times 224$ pixels). In contrast, the evaluation budgets in this project were:
- **Hospital 1 (Development Cohort):** 64 patches per WSI across 5 slides ($N = 320$ patches).
- **Hospital 2 (Target Cohort):** 128 patches per WSI across 10 slides ($N = 1,280$ patches; 40 batches of size 32).

#### Scientific Consequences:
1. **Representational Coverage:** Subsampling 128 patches per slide samples less than $1\%$ of the available slide surface area. While uniform random sampling provides an unbiased sample of the slide-level distribution, it may under-sample micro-metastatic foci consisting of only dozens of tumor cells.
2. **Adaptation Dynamics:** With 128 patches per slide and batches of 32, each slide undergoes exactly 4 adaptation updates. Under learning rate $\eta = 10^{-4}$, total parameter displacement per slide was restricted to $\Delta \le 4.01 \times 10^{-5}$. It remains unproven whether adapting over 30,000 patches (nearly 1,000 updates per slide) would compound confirmation bias into complete collapse or reach an asymptotic equilibrium.

---

## 4. Prior Target-Domain Exposure and Data Isolation

### 4.1 Historical Exposure in v1
During the exploratory v1 benchmarking phase, multi-center Camelyon17 data (including Centers 1 and 2) were utilized to evaluate Tent collapse, EATA, and multi-seed baselines across five random initialization seeds. Consequently, the general characteristics of Center 2 were historically known to the research team.

### 4.2 v2 Quarantine and Isolation Protocol
To restore external validation validity in v2:
1. **Zero Target Tuning:** All architectural mechanisms, hyperparameter thresholds ($\tau = 0.90, \epsilon = 0.05, \eta = 10^{-4}$), and monitor calibrations ($W = 50, \min = 20$) were frozen exclusively on **Hospital 1 (Center 1)** development data.
2. **Single Immutable Run:** Hospital 2 (Center 2) was loaded and evaluated exactly once in Stage E2 under a pre-registered frozen configuration (`configs/v2/safeadapt_e2_frozen.yaml`). No hyperparameter adjustments or post-hoc tuning were permitted after inspecting Hospital 2 metrics.

---

## 5. Statistical Power and Unit-of-Analysis Constraints

### 5.1 Slide-Level vs Patch-Level Independence
A pervasive error in digital pathology literature is computing statistical significance across pooled patches ($N = 1,280$), treating patches from the same patient and histology slide as independent observations. Patches from the same slide share:
- Identical formalin fixation and paraffin embedding artifacts.
- Identical microtome section thickness and blade chatter.
- Identical chemical reagent concentrations and staining durations.
- Identical optical scanner calibration and chromatic aberration.

**True Independent Units:** The true independent observational units are the **10 whole-slide images (WSIs 20–29)**. With $N = 10$ slides, statistical power for subtle paired non-inferiority testing is modest. However, because adaptation produced zero positive transfers and degraded F1 and AUROC on the primary challenging slide (Slide 29), the negative transfer finding is robust.

---

## 6. Algorithmic Failure Modes: Confirmation Bias Under Domain Shift

The fundamental algorithmic limitation of unsupervised test-time adaptation identified by this study is **confirmation bias driven by miscalibrated confidence**:

```
Domain Shift (Staining / Scanner Artifacts)
                  |
                  v
Source Model Produces Confident False Positives (p > 0.90)
                  |
                  v
Confidence Gating Fails to Filter Errors (Threshold tau=0.90 Exceeded)
                  |
                  v
Unsupervised Entropy Minimization Updates Model on Erroneous Predictions
                  |
                  v
Model Adapts to Predict False Positive Features with Higher Confidence
                  |
                  v
Diagnostic Performance Degrades on Outlier Slides (e.g., Slide 29 F1: 0.3659 -> 0.3457)
```

Unsupervised loss functions lack ground-truth supervision. When distribution shift causes the model to become confidently wrong, self-adaptation acts as an error amplifier rather than a self-correcting mechanism.

---

## 7. Clinical and Ethical Deployment Boundaries

> [!CAUTION]
> **ABSOLUTE CLINICAL CONTRAINDICATION**  
> UTTA-Med and its constituent test-time adaptation algorithms are strictly experimental research prototypes. **THEY ARE NOT CLEARED, VALIDATED, OR SUITABLE FOR CLINICAL USE, DIAGNOSTIC SCREENING, OR PATIENT TRIAGE.**

### Specific Deployment Hazards:
1. **Unpredictable Single-Patient Degradation:** While aggregate metrics drop only slightly (Macro AUROC $0.9308 \to 0.9299$), individual patient slides can experience substantial loss of diagnostic sensitivity (Slide 29 F1 drop from $0.3659 \to 0.3457$). In oncology, missing micro-metastases directly impacts staging, chemotherapy decisions, and patient survival.
2. **False Sense of Safety from Silent Execution:** SafeAdapt executed with zero alarms and zero numerical errors on Hospital 2. Clinicians and operators could easily misinterpret the absence of software crashes or collapse alarms as evidence of successful adaptation, unaware that tumor boundary precision is actively degrading.
3. **Regulatory Non-Compliance:** Test-time adaptation models violate the standard "locked algorithm" paradigm required by the US FDA and EU MDR for Software as a Medical Device (SaMD). Continuous weight modification during inference poses severe regulatory hurdles for traceability and liability.
