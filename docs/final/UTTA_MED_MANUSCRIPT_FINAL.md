# Empirical Investigation of Uncertainty Gating and Source-Preserving Test-Time Adaptation under Histopathology Domain Shift

**Authors:** UTTA-Med Research Consortium and Scientific Audit Team  
**Affiliations:** Medical Imaging Machine Learning Laboratory & Independent Reproducibility Audit Group  
**Target Venue:** Medical Image Analysis (MedIA) / IEEE Transactions on Medical Imaging (TMI)  
**Date:** October 2026  
**Artifact Repository:** `UTTA-Med` (Canonical Checkpoint SHA-256: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`)

---

## Abstract

Test-time adaptation (TTA) aims to mitigate distribution shift in deep learning models by updating parameters on unlabelled target instances during inference. While successful in natural computer vision benchmarks, unsupervised entropy minimization methods (such as Tent) frequently suffer from representation collapse or error accumulation when applied to medical imaging data. In this work, we present an empirical investigation into uncertainty-gated test-time adaptation and runtime safeguarding across whole-slide histopathology images from the Camelyon17-WILDS benchmark. 

We evaluate two core components: (1) selective filtering of adaptation updates using Monte Carlo (MC) dropout predictive variance versus softmax confidence heuristics, and (2) source-preserving runtime control (SafeAdapt) featuring an explicitly bounded parameter-update mechanism ($L_\infty \le 0.05$), numerical checks, and atomic state rollback. On development data (Hospital 1, Center 1), we find that MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting, exhibiting a Spearman rank correlation of $\rho = -0.9917$ with max-softmax confidence and introducing a $3.5\times$ computational slowdown. 

On an external target domain (Hospital 2, Center 2; 10 whole-slide images, 1,280 patches), evaluated under a pre-adaptation streaming inference protocol with whole-slide boundary resets, unsupervised confidence-gated adaptation degraded diagnostic performance relative to the frozen source model across Macro AUROC ($0.9308 \to 0.9299$), Pooled AUROC ($0.9364 \to 0.9360$), and Pooled F1 ($0.7840 \to 0.7822$). Performance degradation on outlier slides (e.g., Slide 29 F1 dropping from $0.3659 \to 0.3457$) was consistent with confirmation bias. While SafeAdapt strictly constrained parameter displacement ($\Delta \le 4.01 \times 10^{-5} \ll 0.05$) and prevented numerical errors, bounding parameter updates could not convert negative adaptation dynamics into positive diagnostic transfer. Our findings caution against the unmonitored clinical deployment of unsupervised TTA in computational pathology and emphasize the necessity of whole-slide evaluation units, frozen external validation, and computational frugality.

---

## 1. Introduction and Clinical Motivation

Deep neural networks trained for digital pathology diagnosis frequently suffer from performance degradation when deployed across external hospital networks. Variations in tissue preparation, microtome sectioning thickness, hematoxylin and eosin (H&E) staining protocols, and whole-slide image (WSI) scanner optics introduce pervasive domain shifts (Bandi et al., 2018; Koh et al., 2021). Retraining or fine-tuning models on multi-center data requires centralized data aggregation and exhaustive expert annotation, both of which are constrained by patient privacy regulations and pathologist scarcity.

Test-time adaptation (TTA) has emerged as an appealing alternative (Wang et al., 2021). By updating model parameters (typically batch normalization affine parameters $\gamma$ and $\beta$) on incoming unlabelled test batches via unsupervised objectives such as Shannon entropy minimization, TTA aims to adapt autonomously to novel clinical sites without accessing source training data.

### 1.1 The Fragility of Unsupervised Adaptation in Histopathology
Despite early enthusiasm, recent studies highlight severe vulnerabilities when applying TTA to medical diagnosis:
1. **Representation Collapse:** Unconstrained entropy minimization can drive neural network representations to degenerate, single-class solutions (e.g., predicting all tissue patches as tumor or normal) to achieve trivial zero entropy (Niu et al., 2022; Wang et al., 2021).
2. **Confirmation Bias:** In clinical datasets characterized by high class imbalance or staining variation, models can make high-confidence false-positive predictions. Minimizing entropy on these erroneous predictions reinforces model errors, progressively altering decision boundaries.
3. **Computational Inefficiency:** Prior proposals, including the initial v1 formulation of UTTA-Med, advocated Monte Carlo (MC) dropout to estimate epistemic uncertainty and filter ambiguous patches. However, evaluating multiple stochastic forward passes per patch severely limits high-throughput digital pathology workflows, where single WSIs contain tens of thousands of patches.

### 1.2 Scope and Contributions
In this paper, we conduct an independent scientific audit and empirical evaluation of uncertainty-gated TTA and runtime safeguarding. Our contributions are:
1. **Empirical Evaluation of Gating Heuristics:** We compare softmax confidence, MC-dropout predictive variance, and hybrid gating, establishing that MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting while incurring substantial computational latency.
2. **Source-Preserving Runtime Safeguards:** We evaluate SafeAdapt, an explicitly bounded parameter-update mechanism with tested numerical checks, atomic state rollback, and whole-slide boundary resets.
3. **External Target Domain Evaluation:** We present an immutable, four-arm pre-adaptation evaluation on Hospital 2 of the Camelyon17 benchmark, demonstrating that unsupervised adaptation resulted in negative transfer compared to a frozen source baseline.
4. **Methodological Transparency:** We document historical protocol shifts, separate exploratory development from external confirmation, and provide complete machine-readable reproducibility artifacts.

---

## 2. Methods and Controller Architecture

```
Incoming WSI Streaming Batches (x_t)
                  |
                  v
       +--------------------+
       | Pre-Adaptation     |-----> Diagnostic Predictions (y_hat_t)
       | Inference Engine   |       (Simulates online streaming inference)
       +--------------------+
                  |
                  v
       +--------------------+
       | Gating Mechanism   |-----> Drop Low-Confidence Patches (tau = 0.90)
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
       | 2. ||Delta|| > eps |----(Yes)----> Clamp Parameter Vector (eps = 0.05)
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

### 2.2 Explicitly Bounded Parameter Updates and State Rollback
1. **Numerical Gradient Interception:** Before updating parameters, the candidate gradient vector $g_t = \nabla_{\theta_{\mathrm{BN}}} \mathcal{L}_{\mathrm{ent}}$ is audited. If $\exists j \text{ s.t. } g_{t,j} \in \{\operatorname{NaN}, \pm\infty\}$, the step is rejected.
2. **Parameter Displacement Boundary:** We impose a hard $L_\infty$ proximity constraint relative to the source model parameters $\theta_0$:
$$\|\theta_t - \theta_0\|_\infty \le \epsilon, \quad \epsilon = 0.05$$
If an unconstrained update breaches this sphere, parameters are clamped to the boundary $[\theta_0 - \epsilon, \theta_0 + \epsilon]$.
3. **Atomic Rollback:** Host memory maintains a snapshot of $\theta_{t-1}$, optimizer buffers, and batch normalization running statistics. Any rejected update triggers an atomic rollback, ensuring subsequent batches receive uncorrupted state.
4. **WSI-Boundary Invariance Reset:** In clinical pathology, whole slides represent independent biological and histological specimens. To eliminate cross-slide contamination and ensure slide-order invariance, all adapted parameters $\theta_{\mathrm{BN}}$, optimizer momentums, and monitor windows reset to $\theta_0$ at every slide boundary.

---

## 3. Experimental Protocol

### 3.1 Datasets and Cohort Separation
We utilize the Camelyon17-WILDS benchmark (Bandi et al., 2018; Koh et al., 2021), comprising whole-slide lymph node histopathology sections collected across five medical centers in the Netherlands.
- **Source Training Cohort (Centers 0, 3, 4):** Standard supervised pre-training of ResNet-18 models. Checkpoint Seed 42 is frozen as the canonical base model (SHA-256: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`).
- **Development Cohort (Center 1 / Hospital 1):** Five WSIs (`patient_011`, `013`, `015`, `017`, `019`), evaluated over a canonical subsample of 64 patches per slide ($N = 320$ patches). Reserved exclusively for controller calibration, numerical testing, and gating ablation.
- **External Target Cohort (Center 2 / Hospital 2):** Ten WSIs (`patient_020` through `patient_029`), evaluated over 128 patches per slide ($N = 1,280$ patches; 40 batches of size 32). Fully isolated during development; evaluated exactly once under frozen protocol.

### 3.2 Protocol Distinction: Prediction Timing
Crucially, all v2 evaluations adhere to **pre-adaptation prediction timing**. For each incoming batch $t$, predictions $\hat{y}_t = f_{\theta_t}(x_t)$ are logged as the primary diagnostic output *prior* to computing loss or adapting weights. Parameters are then updated to $\theta_{t+1}$ to process subsequent batches within the slide. This simulates an online streaming diagnostic scanner, contrasting with historical v1 protocols which evaluated predictions post-adaptation.

---

## 4. Results and Empirical Analysis

### 4.1 Gating Redundancy and Computational Cost (Hospital 1)
On Hospital 1 development data ($N = 320$ patches across 5 WSIs), we evaluated candidate gating policies:
- **Frozen Source:** AUROC $0.9736$, F1 $0.8632$, Brier $0.082867$, Latency $48.31$s
- **Confidence-Only ($\tau = 0.90$):** AUROC $0.9738$, F1 $0.8596$, Brier $0.082927$, Latency $168.73$s
- **MC-Only ($K = 20$):** AUROC $0.9737$, F1 $0.8596$, Brier $0.083038$, Latency $598.21$s
- **Hybrid Policy ($\tau = 0.65/0.95$):** AUROC $0.9738$, F1 $0.8632$, Brier $0.082953$, Latency $182.17$s

Across these evaluated patches, the Spearman rank correlation between max-softmax confidence and MC predictive variance was $\rho = -0.9917$. MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting, while incurring a $3.55\times$ latency penalty over confidence gating. Note that while this rank correlation is strong, patches are clustered within five WSIs; hence, pooled patch-level significance values are not treated as independent inferential evidence.

### 4.2 Historical v1 Paired Statistical Comparison
In the historical v1 five-seed multi-center benchmark:
- Confidence-Gated TTA achieved mean F1 of $0.831 \pm 0.032$.
- UTTA-Med achieved mean F1 of $0.830 \pm 0.032$.
- Paired comparison: $\Delta = -0.001013$ ($95\%\text{ CI: } [-0.0015, -0.0005]$, nominal $p = 0.0054$, Holm-adjusted $p = 0.0430$, Cohen's $d = -2.454$).

The v1 Holm-adjusted $p=0.0430$ comparison indicates a small but statistically detectable paired F1 difference under the specified analysis. The large effect size ($d = -2.454$) reflects exceptional consistency across the five seeds despite the absolute difference being practically tiny ($\sim 0.001$ F1). This represents a small statistically detectable difference rather than proof of meaningful clinical inferiority.

### 4.3 External Target Cohort Results (Hospital 2)
The four-arm pre-adaptation evaluation on Hospital 2 across 1,280 patches ($10$ WSIs) yielded:

| Evaluation Arm | Macro AUROC (Valid Slides) | Pooled AUROC | Pooled AUPRC | Pooled F1 Score | Expected Calibration Error (ECE) | Brier Score | Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A: Frozen Source** | **0.9308** | **0.9364** | **0.8897** | **0.7840** | **0.0650** | **0.091709** | **223.85s** |
| **Arm B: Confidence TTA** | 0.9299 | 0.9360 | 0.8891 | 0.7822 | 0.0655 | 0.092051 | 354.10s |
| **Arm C: SafeAdapt Stage B** | 0.9300 | 0.9359 | 0.8891 | 0.7822 | 0.0655 | 0.092053 | 389.90s |
| **Arm D: Active SafeAdapt** | 0.9300 | 0.9359 | 0.8891 | 0.7822 | 0.0655 | 0.092053 | 495.85s |

*Primary Target Finding: Unsupervised test-time adaptation resulted in negative transfer across all evaluated metrics. The frozen source model performed best on the target domain.*

### 4.4 Granular Analysis and Confirmation Bias
Disaggregated slide-level evaluation across the 10 target WSIs indicates that the performance drop was localized to challenging sections:
- Normal tissue slides (`slide_020`, `slide_021`) remained stable with zero false positive tumor predictions.
- On standard tumor slides (`slide_022`, `023`, `025`, `028`), adaptation produced minimal changes.
- **Outlier Slide Degradation (`slide_029`):** On Slide 29, the source model generated high-confidence false-positive predictions. Confidence gating accepted these patches ($\tau \ge 0.90$), and subsequent entropy minimization was followed by a drop in slide F1 from $0.3659$ to $0.3457$ and worsening of Brier score from $0.3247$ to $0.3279$.

While Slide 29's performance degradation is consistent with confirmation bias, our experiments do not establish a strict causal mechanism, which would require tracing individual misclassified updates into subsequent inference steps.

---

## 5. Runtime Safeguards and Numerical Telemetry

Safeguard telemetry recorded across all 40 streaming evaluation batches on Hospital 2 confirms:
- **Numerical Interceptions:** 0 `NaN`/`Inf` events observed.
- **Displacement Boundary Adherence:** Maximum observed parameter displacement was $L_\infty = 4.01 \times 10^{-5}$, well within the allowed boundary ($\epsilon = 0.05$).
- **Collapse Alarm Evaluations:** 40 eligible evaluations produced 0 alarms.

SafeAdapt functioned as an explicitly bounded parameter-update mechanism with tested numerical checks and atomic rollback behavior. However, zero observed alarms under benign micro-updates ($\Delta \approx 4 \times 10^{-5}$) demonstrates a low false-alarm rate under nominal streaming, rather than proof of clinical collapse detection. Furthermore, bounding parameter drift prevents catastrophic representation runaway, but cannot convert a negative adaptation trajectory into positive diagnostic generalization.

---

## 6. Discussion and Limitations

### 6.1 Independent Units of Analysis: WSIs vs Patches
In computational pathology, patches sampled from the same whole slide share staining, illumination, and biological characteristics. Reporting statistical significance at the patch level ($N = 1,280$) introduces pseudo-replication. The true independent observational units are the 10 whole-slide images. Across 10 slides, adaptation caused neutral or negative performance changes, confirming the absence of positive transfer.

### 6.2 Sampling Budget and Subsampling Constraints
Evaluation in Stage E2 was conducted on 128 patches per slide ($1,280$ patches total), compared to 64 patches per slide in development. While this budget enabled thorough streaming evaluation across all slides on standard computational infrastructure, clinical deployment entails processing 20,000 to 50,000 patches per slide. Further investigation is required to determine whether massive patch volumes would compound drift or reach an asymptotic equilibrium.

### 6.3 Ethical and Clinical Deployment Boundaries
UTTA-Med and unsupervised test-time adaptation algorithms are experimental research prototypes and are not suitable for clinical use, diagnostic screening, or patient triage. An algorithm that degrades sensitivity on outlier specimens poses clinical risk in oncology.

---

## 7. Conclusion

This empirical investigation provides a scientifically transparent evaluation of test-time adaptation in computational pathology:
1. Monte Carlo dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting while substantially increasing latency.
2. Unsupervised entropy minimization resulted in negative transfer on the external target hospital under pre-adaptation streaming evaluation.
3. SafeAdapt provided an explicitly bounded parameter-update mechanism with tested numerical checks and atomic rollback behavior, but constraining parameter drift cannot cure underlying confirmation bias.
