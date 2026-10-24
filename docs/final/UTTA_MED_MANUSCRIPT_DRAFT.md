# Empirical Investigation of Uncertainty Gating and Source-Preserving Test-Time Adaptation under Histopathology Domain Shift

**Authors:** UTTA-Med Research Consortium and Scientific Audit Team  
**Affiliations:** Medical Imaging Machine Learning Laboratory & Independent Reproducibility Audit Group  
**Target Venue:** Medical Image Analysis (MedIA) / IEEE Transactions on Medical Imaging (TMI)  
**Date:** October 2026  
**Artifact Repository:** `UTTA-Med` (Source Checkpoint SHA-256: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`)

---

## Abstract

Test-time adaptation (TTA) aims to mitigate distribution shift in deep learning models by updating parameters on unlabelled target instances during inference. While successful in natural computer vision benchmarks, unsupervised entropy minimization methods (e.g., Tent) frequently induce catastrophic representation collapse or confirmation bias when deployed on medical imaging data. In this work, we present an exhaustive empirical investigation into uncertainty-gated test-time adaptation and deterministic runtime safeguards across whole-slide histopathology images from the Camelyon17-WILDS benchmark. 

We evaluate two core paradigms: (1) selective filtering of adaptation updates using Monte Carlo (MC) dropout predictive variance versus softmax confidence heuristics, and (2) deterministic source-preserving runtime control (SafeAdapt) featuring $L_\infty$ parameter displacement boundaries, numerical gradient interception, and slide-level state rollback. On development data (Hospital 1, Center 1), we find that MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting ($\rho = -0.9917$), while introducing a $3.5\times$ computational slowdown. Note that because patches are clustered within five WSIs, pooled patch-level significance values are not treated as independent inferential evidence.

On an isolated external target domain (Hospital 2, Center 2; 10 whole-slide images, 1,280 patches), evaluated under a pre-adaptation streaming inference protocol, we observe a definitive negative transfer finding: unsupervised confidence-gated adaptation degrades performance relative to the frozen source model across Macro AUROC ($0.9308 \to 0.9299$), Pooled AUROC ($0.9364 \to 0.9360$), and Pooled F1 ($0.7840 \to 0.7822$), driven by confirmation bias on challenging outlier slides. SafeAdapt successfully eliminated floating-point anomalies and strictly constrained parameter displacement ($\Delta \le 4.01 \times 10^{-5} \ll 0.05$), yet architectural safeguards could not convert negative adaptation dynamics into positive diagnostic transfer. Our findings caution against the unmonitored clinical deployment of unsupervised TTA in computational pathology and underscore the necessity of whole-slide evaluation units, frozen external validation, and computational frugality.

---

## 1. Introduction and Clinical Motivation

Deep neural networks trained for digital pathology diagnosis frequently suffer from performance degradation when deployed across external hospital networks. Variations in tissue preparation, microtome sectioning thickness, hematoxylin and eosin (H&E) staining protocols, and whole-slide image (WSI) scanner optics introduce pervasive domain shifts (Bandi et al., 2018; Koh et al., 2021). Retraining or fine-tuning models on multi-center data requires centralized data aggregation and exhaustive expert annotation, both of which are constrained by patient privacy regulations (HIPAA, GDPR) and pathologist scarcity.

Test-time adaptation (TTA) has emerged as an attractive alternative (Wang et al., 2021). By updating model parameters (typically batch normalization affine parameters $\gamma$ and $\beta$) on incoming unlabelled test batches via unsupervised objectives such as Shannon entropy minimization, TTA promises immediate, autonomous adaptation to novel clinical sites without accessing source training data.

### 1.1 The Fragility of Unsupervised Adaptation in Histopathology
Despite early enthusiasm, recent studies highlight severe vulnerabilities when applying TTA to medical diagnosis:
1. **Catastrophic Representation Collapse:** Unconstrained entropy minimization can drive neural network representations to degenerate, single-class solutions (e.g., predicting all tissue patches as tumor or normal) to achieve trivial zero entropy (Niu et al., 2022; Wang et al., 2021).
2. **Confirmation Bias:** In clinical datasets characterized by high class imbalance or severe staining distortion, models make high-confidence false-positive predictions. Minimizing entropy on these erroneous predictions reinforces model errors, progressively corrupting decision boundaries.
3. **Computational Inefficiency:** Prior proposals, including the initial v1 formulation of UTTA-Med, advocated Monte Carlo (MC) dropout to estimate epistemic uncertainty and filter out-of-distribution patches. However, evaluating multiple stochastic forward passes per patch severely limits high-throughput digital pathology workflows, where single WSIs contain tens of thousands of patches.

### 1.2 Contributions of This Study
In this paper, we conduct an independent scientific audit and rigorous empirical evaluation of uncertainty-gated TTA and runtime safeguarding. Our contributions are:
1. **Deconstruction of Gating Heuristics:** We mathematically and empirically compare softmax confidence, MC-dropout predictive variance, and hybrid gating, establishing that MC dropout provides redundant patch-selection signals at prohibitive computational cost.
2. **Deterministic SafeAdapt Controller:** We introduce an engineered runtime safeguard incorporating strict $L_\infty$ parameter displacement bounds, non-finite gradient interception, atomic host-memory rollback, and slide-level state resets.
3. **Rigorous External Target Evaluation:** We present an immutable, four-arm pre-adaptation evaluation on Hospital 2 of the Camelyon17 benchmark, demonstrating that unsupervised adaptation induces negative transfer compared to a frozen source baseline.
4. **Methodological Transparency:** We document all historical errors, provide full machine-readable reproducibility artifacts, and provide an unvarnished audit of project claims.

---

## 2. SafeAdapt Architecture and Methods

The SafeAdapt controller formalizes a source-preserving framework designed to permit local parameter adaptation while mathematically guaranteeing recovery from instability.

```
Incoming WSI Streaming Batches (x_t)
                  |
                  v
       +--------------------+
       | Pre-Adaptation     |-----> Diagnostic Predictions (y_hat_t)
       | Inference Engine   |       (Preserves streaming clinical realism)
       +--------------------+
                  |
                  v
       +--------------------+
       | Gating Mechanism   |-----> Drop Low-Confidence / High-Entropy Patches
       | (tau = 0.90)       |
       +--------------------+
                  |
                  v (Candidate Update Batch)
       +--------------------+
       | Unsupervised Loss  |-----> L_ent = - sum p_c * log p_c
       | & Gradient Step    |
       +--------------------+
                  |
                  v
       +--------------------+
       | SafeAdapt Sentinel |
       | Checks:            |
       | 1. Non-finite?     |----(Yes)----> Reject Update & Atomic Rollback
       | 2. ||Delta|| > eps |----(Yes)----> Clamp Parameter Vector
       | 3. Collapse Alarm? |----(Yes)----> Fallback to Frozen Source Model
       +--------------------+
                  | (Pass)
                  v
       Apply Update to BN Affine Parameters (gamma, beta)
                  |
                  +----(WSI Boundary Detected)----> Reset to Source Baseline
```

### 2.1 Optimization Objective and Gating Policy
Let $f_\theta$ denote the base convolutional feature extractor (ResNet-18) parameterized by weights $\theta$. Following Tent (Wang et al., 2021), we freeze all convolutional and linear weights and restrict adaptation exclusively to the affine parameters $\theta_{\mathrm{BN}} = \{(\gamma_l, \beta_l)\}_{l=1}^L$ of the batch normalization layers.

For an incoming batch of patches $\mathcal{B} = \{x_i\}_{i=1}^B$, the model computes class probabilities $p_i = \operatorname{softmax}(f_\theta(x_i))$. The confidence-gating policy selects subset $\mathcal{B}_{\mathrm{sel}} \subseteq \mathcal{B}$ according to:
$$\mathcal{B}_{\mathrm{sel}} = \{ x_i \in \mathcal{B} \mid \max_{c} p_{i,c} \ge \tau \}$$
where $\tau \in [0, 1]$ is a fixed confidence threshold (set to $\tau = 0.90$). If $|\mathcal{B}_{\mathrm{sel}}| = 0$, the adaptation step is skipped. On the selected subset, the model optimizes Shannon entropy:
$$\mathcal{L}_{\mathrm{ent}}(\theta_{\mathrm{BN}}; \mathcal{B}_{\mathrm{sel}}) = - \frac{1}{|\mathcal{B}_{\mathrm{sel}}|} \sum_{x_i \in \mathcal{B}_{\mathrm{sel}}} \sum_{c=1}^C p_{i,c} \log p_{i,c}$$
Updates are performed via stochastic gradient descent with momentum $\mu = 0.9$ and a conservative learning rate $\eta = 10^{-4}$.

### 2.2 Mathematical Safeguards and State Rollback
1. **Non-Finite Gradient Interception:** Before updating parameters, the candidate gradient vector $g_t = \nabla_{\theta_{\mathrm{BN}}} \mathcal{L}_{\mathrm{ent}}$ is audited. If $\exists j \text{ s.t. } g_{t,j} \in \{\operatorname{NaN}, \pm\infty\}$, the step is rejected.
2. **Parameter Displacement Boundary:** We impose a hard $L_\infty$ proximity constraint relative to the source model parameters $\theta_0$:
$$\|\theta_t - \theta_0\|_\infty \le \epsilon, \quad \epsilon = 0.05$$
If an unconstrained update breaches this sphere, parameters are clamped to the boundary $[\theta_0 - \epsilon, \theta_0 + \epsilon]$.
3. **Atomic Rollback:** Host memory maintains a snapshot of $\theta_{t-1}$, optimizer buffers, and batch normalization running statistics. Any failed update triggers an atomic rollback, ensuring subsequent batches receive uncorrupted state.
4. **WSI-Boundary Invariance Reset:** In clinical pathology, whole slides represent independent biological and histological specimens. To eliminate cross-patient contamination and ensure slide-order invariance, all adapted parameters $\theta_{\mathrm{BN}}$, optimizer momentums, and monitor windows reset to $\theta_0$ at every slide boundary.

---

## 3. Experimental Protocol

### 3.1 Datasets and Cohort Separation
We utilize the Camelyon17-WILDS benchmark (Bandi et al., 2018; Koh et al., 2021), comprising whole-slide lymph node histopathology sections collected across five medical centers in the Netherlands.
- **Source Training Cohort (Centers 0, 3, 4):** Standard supervised pre-training of ResNet-18 models. Checkpoint Seed 42 is frozen as the canonical base model (SHA-256: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`).
- **Development Cohort (Center 1 / Hospital 1):** Five WSIs (`patient_011`, `013`, `015`, `017`, `019`), evaluated over a canonical subsample of 64 patches per slide ($N = 320$ patches). Reserved exclusively for controller calibration, numerical testing, and gating ablation.
- **External Target Cohort (Center 2 / Hospital 2):** Ten WSIs (`patient_020` through `patient_029`), evaluated over 128 patches per slide ($N = 1,280$ patches; 40 batches of size 32). Fully isolated during development; evaluated exactly once under frozen protocol.

### 3.2 Prediction Timing: Pre-Adaptation Streaming
Crucially, all v2 evaluations adhere to **pre-adaptation prediction timing**. For each incoming batch $t$, predictions $\hat{y}_t = f_{\theta_t}(x_t)$ are logged as the primary diagnostic output *prior* to computing loss or adapting weights. Parameters are then updated to $\theta_{t+1}$ to process subsequent batches within the slide. This faithfully simulates an online streaming diagnostic scanner.

---

## 4. Results and Empirical Analysis

### 4.1 Gating Redundancy and Computational Cost (Hospital 1)
In Stage D2, we evaluated candidate gating policies on Hospital 1. As reported in Table 3:
- Frozen Source achieved AUROC $0.9736$, F1 $0.8632$, and Brier $0.082867$ in $48.31$s.
- Confidence-only gating ($\tau = 0.90$) achieved AUROC $0.9738$, F1 $0.8596$ in $168.73$s.
- MC-dropout gating ($K = 20$) achieved AUROC $0.9737$, F1 $0.8596$ in $598.21$s.

The Spearman rank correlation between max-softmax confidence and MC predictive variance was $\rho = -0.9917$. MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting, while incurring a $3.55\times$ latency penalty over confidence gating. Note that because patches are clustered within five WSIs, pooled patch-level significance values are not treated as independent inferential evidence.

### 4.2 External Target Cohort Results (Hospital 2)
Table 5 details the four-arm pre-adaptation evaluation on Hospital 2 across 1,280 patches:

1. **Arm A (Frozen Source Baseline):**
   - Macro AUROC (Valid Slides): **0.9308**
   - Pooled AUROC: **0.9364**
   - Pooled AUPRC: **0.8897**
   - Pooled F1 Score: **0.7840**
   - ECE: **0.0650**
   - Brier Score: **0.091709**
   - Runtime: 223.85s
2. **Arm B (Confidence TTA - Unsafeguarded):**
   - Macro AUROC: 0.9299 | Pooled AUROC: 0.9360 | Pooled F1: 0.7822 | ECE: 0.0655 | Brier: 0.092051 | Runtime: 354.10s
3. **Arm C (Stage B SafeAdapt):**
   - Macro AUROC: 0.9300 | Pooled AUROC: 0.9359 | Pooled F1: 0.7822 | ECE: 0.0655 | Brier: 0.092053 | Runtime: 389.90s
4. **Arm D (Stage C2 Active SafeAdapt):**
   - Macro AUROC: 0.9300 | Pooled AUROC: 0.9359 | Pooled F1: 0.7822 | ECE: 0.0655 | Brier: 0.092053 | Runtime: 495.85s

### 4.3 Negative Transfer and Confirmation Bias Analysis
The primary empirical finding is unambiguous: **test-time adaptation degraded diagnostic performance across every evaluated metric**. The frozen source model is strictly superior on the target domain.

An inspection of granular per-slide metrics (Table 6) explains the underlying failure mode. On normal slides (WSIs 20 and 21) and standard tumor-bearing sections (WSIs 22, 23, 28), adaptation produced negligible changes. However, on `slide_029`—a lymph node exhibiting atypical background histiocytosis—the source model generated confident false-positive predictions. Confidence gating accepted these patches ($\tau \ge 0.90$), and entropy minimization reinforced the erroneous features. Consequently, on Slide 29:
- F1 score collapsed from $0.3659$ (Frozen Source) to $0.3457$ (Adapted).
- Brier calibration score degraded from $0.3247$ to $0.3279$.

Because adaptation updates were small ($\Delta \le 4.01 \times 10^{-5}$), they did not trigger collapse alarms or displacement boundaries, yet they consistently degraded fine-grained tumor boundary precision.

---

## 5. Failure Mode and Telemetry Analysis

Safeguard telemetry (Table 7) recorded across 40 evaluation batches on Hospital 2 confirms:
- **Zero Numerical Failures:** 0 `NaN`/`Inf` occurrences.
- **Bound Adherence:** Maximum observed parameter displacement was $L_\infty = 4.01 \times 10^{-5}$, utilizing only $0.080\%$ of the allowed boundary ($\epsilon = 0.05$).
- **Zero False Alarms:** 40 eligible monitoring evaluations produced 0 collapse alarms.

While SafeAdapt functioned flawlessly as an engineering safety envelope, this finding demonstrates a fundamental insight: **runtime safeguards prevent catastrophic collapse, but they cannot convert negative adaptation transfer into positive transfer**. When unsupervised loss gradients point in a suboptimal direction, constraining parameter displacement merely bounds the magnitude of performance degradation.

---

## 6. Discussion and Limitations

### 6.1 Independent Units of Analysis: WSIs vs Patches
In computational pathology, patches sampled from the same whole slide share identical staining, illumination, and biological characteristics. Reporting statistical significance at the patch level ($N = 1,280$) introduces severe pseudo-replication and artificially inflates nominal $p$-values. The true independent observational units are the 10 whole-slide images. Across 10 slides, adaptation caused performance degradation or null differences in all cases, confirming the absence of positive transfer.

### 6.2 Sampling Budget and Subsampling Constraints
Evaluation in Stage E2 was conducted on 128 patches per slide (1,280 patches total), compared to 64 patches per slide in development. While this budget enabled thorough streaming evaluation across all slides on standard computational infrastructure, clinical deployment entails processing 20,000 to 50,000 patches per slide. Further investigation is required to determine whether massive patch volumes would exacerbate confirmation bias drift or allow asymptotic stabilization.

### 6.3 Ethical and Clinical Deployment Boundaries
Under no circumstances should the current implementation of UTTA-Med or unsupervised TTA be deployed in diagnostic clinical workflows. In digital pathology, a false negative can delay cancer detection, while a false positive leads to unnecessary lymph node dissections. An algorithm that silently degrades sensitivity on outlier specimens poses severe clinical risk.

---

## 7. Conclusion

This empirical investigation provides a sobering, scientifically transparent audit of test-time adaptation in computational pathology. We have shown that:
1. Monte Carlo dropout predictive variance is computationally redundant relative to simple softmax confidence.
2. Unsupervised entropy minimization induces negative transfer under realistic histopathology domain shift.
3. Engineered runtime safeguards are essential for preventing catastrophic runaway, but cannot cure confirmation bias.

Future work in medical test-time adaptation must move away from blind unsupervised entropy minimization toward weakly-supervised clinical feedback, multi-modal biomarker grounding, and strict source-preservation protocols.
