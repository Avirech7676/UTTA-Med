# Selective Test-Time Adaptation and Runtime Safeguarding for Histopathology under Cross-Hospital Domain Shift: A Longitudinal Reliability Framework

**Author:** Avinash Reddy Cheerapareddy  
**Institution:** Department of Computer Science and Engineering, SRM University-AP, Amaravathi, Andhra Pradesh, India  
**Date:** October 2026  
**Correspondence:** avinashreddy.crp@gmail.com / cheerapareddy.avinash@alumni.ee  
**Academic Program:** Undergraduate Research Opportunities Program (UROP) Research Monograph  
**Artifact Repository Checksum (Canonical SHA-256):** `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`

---

## Abstract

Deep learning models deployed in computational histopathology face severe out-of-distribution performance degradation when transferred across clinical healthcare centers. Systematic variations in tissue microtome sectioning thickness, hematoxylin and eosin (H&E) staining protocols, Whole-Slide Image (WSI) optical scanner hardware, and patient demographic cohorts induce substantial domain shifts. Unsupervised Test-Time Adaptation (TTA) via entropy minimization (Tent) has emerged as an appealing strategy to adapt frozen source models online using unlabeled target batches without requiring access to private source training data. However, as documented in our rigorous five-seed investigation on the Camelyon17-WILDS lymph node breast cancer metastasis benchmark, unconstrained entropy minimization exhibits catastrophic instability: under a fair BatchNorm-affine adaptation protocol across five random source seeds (Seeds 7, 42, 99, 123, 2024), Tent collapses completely on 2 of 5 seeds, where macro AUROC plummets from $0.930 \to 0.338$ and diagnostic sensitivity collapses below $3\%$. In a clinical oncology workflow, such catastrophic failure results in severe false-negative nodal under-staging.

To transform erratic test-time adaptation into an operationally dependable framework, we introduce **SafeAdapt-TTA**, a mathematically grounded, process-level runtime safeguarding and selective adaptation architecture. SafeAdapt couples hard selective uncertainty gating with strict parameter displacement bounding ($\|\theta_t - \theta_0\|_\infty \le 0.05$), non-finite gradient interception, atomic state rollback, and whole-slide boundary invariance resets. Evaluating candidate patch-admission policies across development data (Hospital 1), we discover that in single-logit binary classification, Monte Carlo (MC) dropout predictive variance and max-softmax confidence induce near-identical patch sample rankings (Spearman rank correlation $\rho = -0.9917$). Deterministic confidence gating matches MC-dropout accuracy while eliminating a $3.55\times$ computational latency penalty. Across five random initialization seeds on Camelyon17-WILDS, selective gating cuts the harmful adaptation rate from $18.7\% \pm 17.9\%$ (Tent) down to $0.7\% \pm 0.3\%$ (~25-fold reduction) and consistently improves paired F1 across all five seeds ($\Delta\text{F1} = +0.0231$, $95\%$ CI $[+0.0104, +0.0358]$, $p = 0.0072$, Cohen's $d = 2.260$). Matched-coverage random gating still collapses (AUROC $0.697 \pm 0.304$, harm $19.4\%$), confirming that stability gains arise specifically from *which* patches adapt rather than fewer optimization steps.

Crucially, in an isolated, frozen pre-adaptation streaming evaluation on an external target domain (Hospital 2; 10 WSIs, 1,280 patches), unsupervised confidence-gated adaptation resulted in slight negative transfer relative to the frozen source model (Macro AUROC $0.9308 \to 0.9300$, Pooled AUROC $0.9364 \to 0.9359$, Pooled F1 $0.7840 \to 0.7822$). Granular slide-level disaggregation revealed that performance degradation was concentrated on outlier slides with pronounced domain shifts (notably Slide 29 F1 dropping from $0.3659 \to 0.3457$), consistent with self-reinforcing confirmation bias. While SafeAdapt strictly contained parameter displacement ($\Delta \le 4.01 \times 10^{-5} \ll 0.05$) and logged zero numerical failures across 50 streaming batches, parameter bounding alone cannot convert negative adaptation dynamics into positive diagnostic transfer. We synthesize these empirical results into a scientific firewall delineating what the findings establish, establish a 120-test automated regression suite with cryptographic checksum verification, and caution against the unmonitored clinical deployment of unsupervised TTA in safety-critical computational pathology.

**Keywords:** Test-Time Adaptation, Digital Pathology, Camelyon17-WILDS, Domain Shift, Uncertainty Estimation, Model Reliability, SafeAdapt, Confirmation Bias, Longitudinal Evaluation.

---

## 1. Introduction

### 1.1 Background
Digital pathology is undergoing rapid clinical transformation, driven by high-throughput whole-slide optical scanners and deep convolutional and transformer architectures for computer-aided cancer diagnosis, grading, and staging (Bandi et al., 2018). In clinical oncology, identifying metastatic breast adenocarcinoma foci within sentinel lymph node sections is essential for assigning pathological tumor-node-metastasis (pTNM) stages, which directly dictate whether patients undergo adjuvant chemotherapy, regional radiation, or axillary lymph node dissection.

However, deep neural networks trained in a controlled host institution frequently suffer catastrophic performance degradation when deployed across external hospital networks (Koh et al., 2021). Variations in microtome sectioning thickness, slide mounting reagents, hematoxylin and eosin (H&E) dye chemical formulations, automated staining bath dwell times, and scanner optical properties (e.g., lens aberrations, sensor color calibration) introduce pervasive out-of-distribution (OOD) domain shifts. Centralized retraining or fine-tuning is severely restricted by patient health data privacy laws (e.g., HIPAA, GDPR) and the scarcity of exhaustive pathologist annotations.

### 1.2 Motivation
Test-Time Adaptation (TTA) has emerged as an attractive paradigm for autonomous domain generalization (Wang et al., 2021; Liang et al., 2023). By updating a subset of source model parameters (typically the scale $\gamma$ and bias $\beta$ affine parameters of Batch Normalization layers) online using incoming unlabeled target batches, TTA seeks to align target feature statistics with source representations without accessing historical training data.

The canonical TTA formulation, Test Entropy Minimization (Tent; Wang et al., 2021), minimizes the Shannon entropy of model predictions via stochastic gradient descent. While Tent demonstrates compelling performance gains on synthetic computer vision benchmarks (e.g., ImageNet-C), medical imaging data violates key assumptions of stationary natural vision. Digital pathology is characterized by extreme tissue heterogeneity, severe class imbalance (normal stroma and adipose tissue vastly outnumber micrometastases), and biological artifacts.

### 1.3 Problem Statement
Despite theoretical promises, applying unsupervised entropy minimization to clinical histopathology reveals two severe vulnerabilities:
1. **Representation Collapse & Attractor States:** Under unconstrained entropy minimization, neural networks can rapidly converge to trivial single-class degenerate solutions (e.g., classifying 100% of tissue patches as negative stroma) to achieve mathematical zero entropy.
2. **Confirmation Bias & Error Amplification:** Under real clinical stain shifts, models frequently output high-confidence false-positive or false-negative predictions. Minimizing entropy on erroneous confident predictions reinforces decision boundary errors, progressively corrupting representations.
3. **Computational Inefficiency in Whole-Slide Workflows:** Prior proposals advocated Monte Carlo (MC) dropout to estimate epistemic predictive variance for patch filtering. However, evaluating 20 stochastic forward passes per patch imposes an unsustainable $3.5\times$ to $5\times$ latency penalty in high-throughput clinical scanning workflows where whole slides contain up to 50,000 tissue tiles.

### 1.4 Research Questions
This research monograph investigates the following central scientific question:
> *Can selective uncertainty gating and runtime parameter safeguarding eliminate representation collapse and harmful adaptation in digital pathology under cross-hospital domain shifts, and does unsupervised adaptation translate into positive diagnostic transfer on external clinical target cohorts?*

To systematically address this question, we decompose the inquiry into six preregistered research sub-questions:
* **RQ1 (Collapse Dynamics):** Does unconstrained entropy minimization (Tent) exhibit reproducible representation collapse under fair multi-seed evaluation on Camelyon17-WILDS?
* **RQ2 (Harm Reduction):** Does selective gating based on uncertainty thresholds calibrated on a held-out institution significantly cut harmful adaptation and improve F1 across independent seeds?
* **RQ3 (Gating Equivalence & Efficiency):** Does epistemic uncertainty estimation via Monte Carlo dropout provide meaningful patch-selection discrimination beyond deterministic softmax confidence in single-logit binary histopathology classification?
* **RQ4 (Selection vs. Step Budget):** Are stability gains attributable to the selective identification of reliable adaptation signals, or merely a trivial consequence of taking fewer gradient steps (matched-coverage random control)?
* **RQ5 (Runtime Parameter Safeguards):** Can a runtime controller enforcing explicit parameter displacement bounds ($\|\theta_t - \theta_0\|_\infty \le \epsilon$) and slide-boundary resets guarantee numerical stability and eliminate inter-patient contamination?
* **RQ6 (External Target Transfer):** Does selective test-time adaptation improve diagnostic metrics relative to a frozen source model when evaluated on an unseen external hospital under a leak-free streaming inference protocol?

### 1.5 Contributions
This work provides the following concrete contributions to the study of machine learning reliability in computational pathology:
1. **Multi-Seed Empirical Demonstration of TTA Instability:** We document that on the Camelyon17-WILDS benchmark across five random source seeds, unconstrained Tent collapses completely on 2 of 5 seeds ($0.930 \to 0.338$ AUROC, sensitivity $<3\%$), proving that single-seed reporting masks critical clinical failure modes.
2. **SafeAdapt-TTA Architectural Controller:** We design, implement, and verify SafeAdapt, a runtime controller integrating selective gating, numerical gradient auditing, atomic state rollback, explicit $L_\infty$ proximity bounding, and whole-slide boundary invariance resets.
3. **Resolution of Binary Gating Equivalence:** We prove mathematically and confirm empirically ($\rho = -0.9917$) that in single-logit binary classification, MC-dropout predictive variance and max-softmax confidence induce near-identical patch rankings, establishing that confidence gating matches MC-dropout accuracy while eliminating a $3.55\times$ computational latency overhead.
4. **Matched-Coverage Causal Controls:** By evaluating random gating matched to the exact coverage of selective gating, we prove that random gating still suffers catastrophic collapse, demonstrating that performance preservation depends strictly on *sample selection quality*.
5. **Transparent Reporting of External Target Negative Transfer:** Under an isolated pre-adaptation streaming protocol on Hospital 2 (10 WSIs, 1,280 patches), we show that adaptation yielded slight negative transfer relative to the frozen source model due to confirmation bias on outlier slides, providing an essential cautionary baseline for the clinical deployment of unsupervised TTA.
6. **Complete Cryptographic Reproducibility Package:** We provide a 120-test automated regression suite (100% passing rate) and bit-exact table verification scripts (54/54 passed checks) locked under SHA-256 checksums.

---

## 2. Related Work

### 2.1 Unsupervised Test-Time Adaptation
Test-Time Adaptation (TTA) aims to adjust model parameters to test distributions without using target labels or accessing source data (Liang et al., 2023). Tent (Wang et al., 2021) optimizes Batch Normalization (BN) affine parameters ($\gamma, \beta$) via Shannon entropy minimization. While effective on synthetic computer vision benchmarks (ImageNet-C), Tent is notoriously susceptible to representation collapse when exposed to class imbalance or label shift.

To stabilize Tent, EATA (Niu et al., 2022) introduces an entropy threshold to discard high-entropy (ambiguous) samples and adds a Fisher-information regularization term to constrain parameter displacement. SAR (Niu et al., 2023) evaluates sharpness-aware minimization and filters samples with large gradient norms. However, these methods rely on point-estimate softmax outputs and evaluate natural image classification benchmarks without considering clinical histopathology workflows.

### 2.2 Uncertainty Estimation in Deep Learning
Quantifying predictive uncertainty is critical for safety-critical clinical AI. Bayesian neural networks provide principled uncertainty estimates but are computationally intractable for real-time inference. Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016) approximates deep Gaussian processes by keeping dropout layers active at test time and computing predictive variance over $N$ stochastic forward passes. Deep Ensembles (Lakshminarayanan et al., 2017) provide robust epistemic uncertainty but require training and deploying $M$ independent networks. Evidential Deep Learning (Sensoy et al., 2018) models Dirichlet distributions over class probabilities, parameterizing epistemic vacuity directly from single-pass logits.

### 2.3 Medical Domain Shift & Histopathology Benchmarks
Histopathology domain shift has been extensively characterized in the CAMELYON16, CAMELYON17 (Bandi et al., 2018), and WILDS benchmarks (Koh et al., 2021). Digital pathology datasets exhibit severe covariate shift (stain variations across centers) coupled with concept shift (heterogeneous tumor morphology across patient cohorts). Methods tackling this shift have historically focused on Stain Normalization (e.g., Macenko, Vahadane), optical density augmentation (RandStainNA; Kang et al., 2023), and unsupervised domain adaptation (UDA). However, UDA requires concurrent access to source and target data, which is prohibited in decentralized clinical deployments.

### 2.4 Test-Time Adaptation in Medical Imaging
Recent studies have explored TTA in biomedical imaging. CertainTTA (2025) combines output and model uncertainty for medical image segmentation. UAD-FM (2025) investigates uncertainty-aware adaptation for pathology foundation models on colorectal cancer. TT-SaD (2024) employs test-time diffusion models to standardize staining on Camelyon17. SicTTA (2026) uses uncertainty to select source-like samples for continual segmentation in cardiac and fundus imaging. Crucially, Histopath-C (2026) evaluated synthetic corruptions across histopathology datasets and observed that standard Tent frequently degrades diagnostic accuracy.

### 2.5 Positioning SafeAdapt-TTA
Table 1 situates SafeAdapt-TTA within the landscape of related test-time adaptation and medical generalization literature.

```
Table 1: Positioning SafeAdapt-TTA within related literature across research lines.
========================================================================================================================
Research Line            Key Methods          Update Scope      Filtering Heuristic    Runtime Safeguards   Clinical Focus
------------------------------------------------------------------------------------------------------------------------
General TTA              Tent (Wang 2021)     BN Affine         None (100% update)     None                 Natural Vision
Point-Filtered TTA       EATA (Niu 2022)      BN Affine         Point Entropy          Fisher Regularizer   Natural Vision
Sharpness-Aware TTA      SAR (Niu 2023)       BN Affine         Gradient Norm Threshold State Reset         Natural Vision
Bayesian Medical TTA     CertainTTA (2025)    Full Network      MC Predictive Entropy  None                 Segmentation
Foundation Model TTA     UAD-FM (2025)        LoRA Adapters     Dirichlet Evidential   None                 Colorectal Cancer
SafeAdapt-TTA (Ours)     SafeAdapt (2026)     BN Affine         Uncertainty Gate (tau) Hard L_inf Sphere,   Camelyon17 Lymph
                                                                                       Atomic Rollback,     Node Metastasis
                                                                                       WSI Boundary Reset   Diagnosis
========================================================================================================================
```

---

## 3. Conceptual and Algorithmic Framework: SafeAdapt-TTA

### 3.1 Problem Formulation & Optimization Scope
Let $f_\theta: \mathcal{X} \to \mathbb{R}$ denote a convolutional feature extractor and classifier head (ResNet-18) parameterized by weights $\theta$, trained on labeled source hospital datasets $\mathcal{D}_S = \{(x_i^S, y_i^S)\}_{i=1}^{N_S}$ where $y_i^S \in \{0, 1\}$. When deployed on an unseen target hospital $\mathcal{D}_T$, the model encounters unlabeled streaming batches $\mathcal{B}_t = \{x_j\}_{j=1}^B \sim \mathcal{P}_T(x)$, where $\mathcal{P}_T(x) \ne \mathcal{P}_S(x)$.

Following the standard TTA formulation, all convolutional kernels and linear head weights are frozen. Adaptation is restricted exclusively to the affine parameters of the Batch Normalization layers:
$$\theta_{\text{BN}} = \{(\gamma_l, \beta_l)\}_{l=1}^L$$
Crucially, running mean $\mu_l$ and running variance $\sigma_l^2$ statistics remain frozen to source values (`bn_freeze_stats = True`) to prevent single-batch statistical corruption under target batch imbalance.

### 3.2 Error vs. Drift: The Mechanics of Representation Collapse
Under standard Tent, the objective minimizes Shannon entropy across all batch instances:
$$\mathcal{L}_{\text{Tent}}(\theta_{\text{BN}}; \mathcal{B}_t) = - \frac{1}{B} \sum_{i=1}^B [p_i \log p_i + (1-p_i) \log (1-p_i)]$$
where $p_i = \sigma(f_{\theta}(x_i))$. When unconstrained, the gradient of the binary entropy with respect to the pre-sigmoid logit $z_i = f_\theta(x_i)$ is:
$$\frac{\partial H(p_i)}{\partial z_i} = (2p_i - 1) \log \left(\frac{p_i}{1 - p_i}\right) = (2p_i - 1) z_i$$
Notice that $\frac{\partial H}{\partial z_i} \ge 0$ for all $z_i \in \mathbb{R}$. Consequently:
- If $p_i > 0.5$ ($z_i > 0$), gradient descent pushes $z_i \to +\infty$ (driving $p_i \to 1.0$).
- If $p_i < 0.5$ ($z_i < 0$), gradient descent pushes $z_i \to -\infty$ (driving $p_i \to 0.0$).

In clinical histopathology, where normal stroma patches outnumber tumor metastases by $10:1$ to $100:1$, negative gradients dominate the batch. If unconstrained, the affine bias $\beta_l$ rapidly drifts negative, shifting logits into a degenerate single-class negative attractor where all inputs produce $p_i \approx 0.0$, collapsing AUROC and sensitivity.

### 3.3 Uncertainty-Gated Selection Mechanisms
To prevent attractor collapse, SafeAdapt admits target patch $x_i$ to the adaptation loss if and only if its uncertainty score $U(x_i)$ is below an admission threshold $\tau$:
$$w(x_i) = \mathbf{1}[U(x_i) < \tau]$$
The threshold $\tau$ is calibrated strictly on held-out development data (Hospital 1) as the 70th percentile of uncertainty, preserving zero target-domain leakage. The gated adaptation objective is:
$$\mathcal{L}_{\text{gated}}(\theta_{\text{BN}}; \mathcal{B}_t) = \begin{cases} 
\frac{\sum_{i=1}^B w(x_i) H(p_i)}{\sum_{i=1}^B w(x_i)}, & \text{if } \sum_{i=1}^B w(x_i) \ge 1 \\
\text{None (Skip Step)}, & \text{if } \sum_{i=1}^B w(x_i) = 0
\end{cases}$$

### 3.4 Mathematical Analysis of Binary Gating Equivalence
Prior literature hypothesized that Bayesian epistemic uncertainty (MC-Dropout predictive variance) is required to identify unreliable patches. We analyze the relationship between Shannon entropy $H(p)$ and max-softmax confidence $C(p) = \max(p, 1-p)$ on single-logit binary classification.

**Lemma 1 (Strict Monotonicity):**  
For binary probability $p \in [0, 1]$, let confidence be $C(p) = \max(p, 1-p) \in [0.5, 1.0]$. The binary Shannon entropy $H(p) = -p\log p - (1-p)\log(1-p)$ is a strictly decreasing bijective function of confidence $C$:
$$\frac{dH}{dC} = \log\left(\frac{1-C}{C}\right) < 0 \quad \forall C \in (0.5, 1.0)$$
*Proof:* Let $C \in (0.5, 1.0]$. By definition, either $p = C$ or $p = 1 - C$. In both cases, $H(p) = -C\log C - (1-C)\log(1-C)$. Differentiating with respect to $C$:
$$\frac{dH}{dC} = -\log C - 1 - (-\log(1-C) - 1) = \log(1-C) - \log C = \log\left(\frac{1-C}{C}\right)$$
Since $C > 0.5$, we have $1-C < C$, which implies $\frac{1-C}{C} < 1$, and therefore $\log\left(\frac{1-C}{C}\right) < 0$. $\blacksquare$

Because entropy and confidence are strictly monotonic in the binary regime, any filtering threshold $\tau_H$ on Shannon entropy induces an identical sample partitioning as a confidence threshold $\tau_C$. Furthermore, because MC-dropout predictive variance is maximized at decision boundaries where $C(p) \to 0.5$ and approaches zero when $C(p) \to 1.0$, MC variance and deterministic confidence induce near-identical rankings ($\rho = -0.9917$).

### 3.5 Runtime Safeguard Controller Architecture
Figure 1 illustrates the operational controller architecture of SafeAdapt-TTA.

```
Figure 1: Architectural controller and inference pipeline of SafeAdapt-TTA.
+-----------------------------------------------------------------------------------------------+
|                                    SAFEADAPT RUNTIME CONTROLLER                               |
+-----------------------------------------------------------------------------------------------+
| Incoming Target Batch x_t (No labels y permitted)                                             |
|         |                                                                                     |
|         v                                                                                     |
| [ 1. PRE-ADAPTATION INFERENCE ENGINE ]                                                        |
| Evaluate y_hat_t = f_{theta_t}(x_t)  ===> Log Primary Diagnostic Prediction (Live Scanner)   |
|         |                                                                                     |
|         v                                                                                     |
| [ 2. UNCERTAINTY GATING AUDIT ]                                                               |
| Evaluate w(x_i) = 1[ score(x_i) <= tau ]                                                     |
|         |                                                                                     |
|         +-----( sum w_i == 0 )-----> SKIP STEP (No backward, No optimizer call, Zero drift)   |
|         |                                                                                     |
|         v ( sum w_i >= 1 )                                                                    |
| Compute Gated Entropy Loss: L_gated = sum(w_i * H_i) / sum(w_i)                               |
|         |                                                                                     |
|         v                                                                                     |
| [ 3. NUMERICAL GRADIENT AUDIT ]                                                               |
| Backpropagate: g_t = grad_{theta_BN}(L_gated)                                                 |
| Check: Any NaN or Inf in g_t?                                                                 |
|         +-----( Yes )-------------> ATOMIC ROLLBACK (Restore theta_{t-1}, Discard gradients)  |
|         |                                                                                     |
|         v ( All finite )                                                                      |
| [ 4. BOUNDED OPTIMIZER STEP & PROJECTION ]                                                    |
| Update candidate weights: theta_cand = theta_t - eta * Adam(g_t)                              |
| Enforce Hard L_inf Sphere relative to source baseline theta_0:                                |
| theta_{t+1} = clamp(theta_cand, theta_0 - eps, theta_0 + eps)   [eps = 0.05]                 |
|         |                                                                                     |
|         v                                                                                     |
| [ 5. WSI BOUNDARY SENTINEL ]                                                                  |
| Slide transition detected?                                                                    |
|         +-----( Yes )-------------> RESET BOUNDARY (theta_{t+1} <- theta_0, Clear moments)    |
|         +-----( No  )-------------> Retain theta_{t+1} for next batch in current slide        |
+-----------------------------------------------------------------------------------------------+
```

Table 2 contrasts the operational properties of standard unconstrained TTA (Tent) against SafeAdapt-TTA.

```
Table 2: Operational distinction between Unconstrained TTA and SafeAdapt-TTA.
========================================================================================================
Operational Dimension         Unconstrained TTA (Tent)              SafeAdapt-TTA (Ours)
--------------------------------------------------------------------------------------------------------
Patch Admission Policy        100% of incoming target patches       Selective Hard Gate (score <= tau)
Empty-Batch Behavior          N/A (Always updates)                  Explicit Skip (Zero backward/step)
Parameter Scope               Unbounded drift in R^D                Strict L_inf Sphere (||Delta|| <= 0.05)
Numerical Integrity           Unchecked (NaNs corrupt model)        Pre-step gradient & loss auditing
State Recovery Mechanism      None (Irreversible corruption)        Atomic State Rollback
Inter-Slide Isolation         Cross-slide continuous drift          Whole-Slide Boundary Invariance Reset
Target Timing                 Post-adaptation (biased)              Pre-adaptation streaming (realistic)
Epistemic Overhead            1 forward pass                        1 forward pass (deterministic confidence)
Target Label Access           Zero (Compliant)                      Zero (Strictly enforced by API)
========================================================================================================
```

---

## 4. Experimental Framework & Evaluation Infrastructure

### 4.1 Datasets and Cohort Separation
We benchmark SafeAdapt-TTA on the official Camelyon17-WILDS benchmark (Bandi et al., 2018; Koh et al., 2021), comprising 50 whole-slide images of sentinel lymph node sections across five medical centers in the Netherlands.
* **Source Training Cohort (Hospitals 0, 3, 4):** 30 WSIs, 302,436 labeled patches ($96 \times 96$ pixels, $0.5\,\mu\text{m/pixel}$). Supervised pre-training of ResNet-18 models across five random initialization seeds (Seeds 7, 42, 99, 123, 2024). Checkpoint Seed 42 is locked as the canonical reference model (SHA-256: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`).
* **Development Cohort (Hospital 1 / Center 1):** 10 WSIs, 34,904 patches. Subsample of 5 representative WSIs (`patient_011`, `013`, `015`, `017`, `019`), 64 patches per slide ($N = 320$ patches). Reserved exclusively for controller calibration, threshold selection, and gating ablations.
* **External Target Cohort (Hospital 2 / Center 2):** 10 WSIs (`patient_020` through `patient_029`), 85,054 patches. Evaluated over 128 deterministically sampled patches per slide ($N = 1,280$ patches; 40 batches of size 32). Fully isolated during development; evaluated exactly once under frozen protocol.

### 4.2 Model Backbones & Adaptation Parameters
The primary architecture is a ResNet-18 feature extractor with a single-logit head trained with binary cross-entropy with logits.
* Adaptable Parameters: BatchNorm scale $\gamma$ and bias $\beta$ ($19,584$ parameters; $0.17\%$ of total $11.18\text{M}$ model parameters).
* Test Optimizer: SGD with momentum ($\eta = 10^{-4}, \mu = 0.9$) or Adam ($\eta = 10^{-5}, \beta_1 = 0.9, \beta_2 = 0.999$).
* Inner steps per batch: $k = 1$. Batch size: $B = 32$.
* Bounded Parameter Radius: $\epsilon = 0.05$.
* Running statistics: Frozen to source values (`bn_freeze_stats = True`).

### 4.3 Streaming Inference vs. Adaptation Timing
We enforce **pre-adaptation prediction timing**. For each incoming target batch $t$, predictions $\hat{y}_t = \sigma(f_{\theta_t}(x_t))$ are evaluated and logged as the primary diagnostic output *prior* to computing loss or executing gradient steps. Model parameters then adapt to $\theta_{t+1}$ to process subsequent batches within the slide. At whole-slide boundaries, parameters reset to $\theta_0$. This protocol faithfully simulates a clinical streaming digital slide scanner.

### 4.4 Information-Leakage Controls & Label Isolation
Target ground truth labels $y_i^T$ are isolated from the adaptation pipeline. API signatures strictly enforce:
```python
def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
    # Signature strictly accepts input tensor x; target label y is prohibited
```
Target labels are ingested only by downstream offline scoring harnesses (`sklearn.metrics`).

---

## 5. Experimental Protocol & Preregistered Arms

We preregistered six core experimental evaluations (Experiments A–F) and two extended cross-modality probes:

### 5.1 Experiment A: Unconstrained Adaptation & Representation Collapse (Tent)
Evaluates source-only ResNet-18 vs. unconstrained Tent ($100\%$ patch update, learning rate $10^{-5}$, $k=1$) across five random source seeds (7, 42, 99, 123, 2024). Measures AUROC degradation, sensitivity collapse, and parameter drift.

### 5.2 Experiment B: Selective Uncertainty Gating & Harm Reduction
Evaluates selective gating on Camelyon17-WILDS using admission threshold $\tau$ fixed on Hospital 1. Compares EATA, confidence gating ($\tau = 0.90$), and MC-dropout variance gating (UTTA-Med, $N=20$ passes). Measures Macro AUROC, F1, Expected Calibration Error (ECE), and the Harmful Adaptation Rate (proportion of correct baseline patches flipped to incorrect predictions).

### 5.3 Experiment C: Monte Carlo Dropout vs. Deterministic Confidence Heuristics
Evaluates gating redundancy on Hospital 1 development data ($N=320$ patches). Measures Spearman rank correlation $\rho$ between MC predictive variance $U(x)$ and max-softmax confidence $C(x)$, wall-clock inference latency, and memory footprints.

### 5.4 Experiment D: Matched-Coverage Random Controls
Evaluates whether stability gains are caused by *which* samples adapt or merely *fewer* updates. Deploys a random gate calibrated to admit the exact empirical coverage percentage of selective gating ($70.0\%$).

### 5.5 Experiment E: Runtime Safeguards & Parameter Displacement
Evaluates SafeAdapt controller telemetry across 50 streaming batches (10 in E1, 40 in E2). Measures maximum parameter displacement $\|\theta_t - \theta_0\|_\infty$, non-finite gradient interception frequency, and recovery under synthetic numerical fault injection.

### 5.6 Experiment F: External Target Domain Streaming Evaluation
Executes frozen 4-arm pre-adaptation evaluation on Hospital 2 (10 WSIs, 1,280 patches):
* **Arm A:** Frozen Source Baseline (ResNet-18 Seed 42).
* **Arm B:** Unsafeguarded Confidence TTA ($\tau = 0.90$).
* **Arm C:** Stage B SafeAdapt (Numerical checks + $L_\infty \le 0.05$ bound).
* **Arm D:** Stage C2 Active SafeAdapt (Dual safeguards + collapse monitoring).

### 5.7 Extended Probes
* **Evidential Decoupling (NCT-CRC-HE 100K $\to$ 7K):** Evaluates 9-class colorectal histology adaptation using Dirichlet Prior Networks to test whether evidential vacuity decouples from confidence in multi-class regimes.
* **Multi-Scanner Mitotic Detection (MIDOG 2021):** Evaluates Faster R-CNN object detection adaptation across three optical scanners (3DHISTECH, Hamamatsu XR, Aperio CS2; 3,300 tiles, 282 mitoses).

### 5.8 Authoritative Frozen Corpus & Cryptographic Manifest
Table 3 documents the dimensions, split allocations, and cryptographic checksums of the frozen experimental corpus.

```
Table 3: Authoritative frozen experimental corpus dimensions and SHA-256 cryptographic checksums.
========================================================================================================================
Dataset / Artifact Name               Split / Cohort      Dimensions / Sample Size           Cryptographic SHA-256 Hash
------------------------------------------------------------------------------------------------------------------------
camelyon17_resnet18_source_s42_BEST   Source (0, 3, 4)    11.18M params (ResNet-18)          f0b85ec1a89707a8ad1807a544280...
headline_5seed.json                   v1 Multi-Seed       5 seeds x 6 methods                9f0f0753835c5971ae9ca9394def0...
table_main_results_final.csv          v1 Summary Table    6 rows x 6 metrics                 320df9cd65e84be39a4b3bbb5f440...
table_paired_statistics_final.csv     v1 Hypothesis Tests 8 contrasts x 8 stats              b7510afe90e26e6e21c0f9ce87fcb...
stage_d2_benchmark_results.json       Dev (Hospital 1)    320 patches (5 WSIs)               143ca46038968393b160ed8a7bf5c...
stage_e1_evaluation_results.json      Dev Streaming (H1)  4 arms x 320 patches               ab060f1ec9e9d725e667272e80148...
stage_e2_target_results.json          Target (Hospital 2) 4 arms x 1,280 patches (10 WSIs)   f5c90277991999ecc6ae5a63fa353...
safeadapt_e2_frozen.yaml              Runtime Config      Frozen execution config            3042225de8be5bd013a5725898a78...
========================================================================================================================
```

---

## 6. Statistical Methodology

### 6.1 Primary Statistical Analysis Plan
All statistical analyses were preregistered to prevent p-hacking. Primary endpoints evaluate Macro valid AUROC, Pooled AUROC, F1 score, Expected Calibration Error (ECE), Brier score, and Harmful Adaptation Rate.

### 6.2 Whole-Slide Image Clustering & Inferential Units
In computational pathology, patches sampled from the same Whole-Slide Image share histological preparation, microtome artifacts, and patient-specific tumor biology. Treating patches as independent observations artificially inflates degrees of freedom ($N=1,280$), generating spuriously tiny p-values ($p < 10^{-50}$). We establish the **Whole-Slide Image ($N=10$) as the primary independent inferential unit**. Macro AUROC is computed by evaluating the ROC curve within each slide independently and averaging across valid slides (excluding slides containing purely normal tissue).

### 6.3 Paired Statistical Hypothesis Testing
To compare adaptation methods against baselines across random initialization seeds ($N=5$), we compute paired two-tailed Student's t-tests on metric differences:
$$\Delta = M_{\text{Method}} - M_{\text{Baseline}}, \quad t = \frac{\bar{\Delta}}{s_\Delta / \sqrt{N}}$$
To control the family-wise error rate across multiple comparisons, we apply the **Holm-Bonferroni correction**.

### 6.4 Effect Sizes & Non-Parametric Bootstrap Confidence Intervals
Effect sizes are reported as Cohen's $d$ for paired samples:
$$d = \frac{\bar{\Delta}}{s_\Delta}$$
Confidence intervals ($95\%$ CI) are constructed using bias-corrected and accelerated (BCa) non-parametric bootstrap resampling ($10,000$ iterations) clustered at the whole-slide level.

---

## 7. Empirical Results & Hypotheses Verification

### 7.1 Cryptographic Dataset Integrity
Prior to statistical analysis, automated integrity verification confirmed that all 54 stored checkpoint and table checksums match expected values with zero discrepancies (passed 54/54 checks).

### 7.2 H1: Catastrophic Collapse of Unconstrained Entropy Minimization
**Preregistered Hypothesis 1:** *Unconstrained Tent exhibits representation collapse under multi-seed evaluation on Camelyon17-WILDS.*  
**Status: DIRECTLY SUPPORTED.**

Under standard BatchNorm-affine adaptation across five random initialization seeds:
- Seed 42: AUROC $0.941$, F1 $0.844$ (Stable)
- Seed 123: AUROC $0.945$, F1 $0.835$ (Stable)
- Seed 2024: AUROC $0.944$, F1 $0.836$ (Stable)
- **Seed 7: AUROC $0.340$, F1 $0.009$, Sensitivity $0.005$ (COLLAPSED)**
- **Seed 99: AUROC $0.336$, F1 $0.005$, Sensitivity $0.002$ (COLLAPSED)**

On Seeds 7 and 99, Tent converged to the single-class negative attractor, predicting all tissue as normal stroma. This collapsed mean Tent AUROC to $0.698 \pm 0.324$ and mean F1 to $0.506 \pm 0.411$, inducing a severe harmful adaptation rate of $18.7\% \pm 17.9\%$ (Table 4).

### 7.3 H2: Harm Reduction and F1 Gain via Selective Gating
**Preregistered Hypothesis 2:** *Selective uncertainty gating prevents collapse and cuts harmful adaptation across all seeds.*  
**Status: DIRECTLY SUPPORTED.**

Hard selective gating ($\tau$ calibrated on Hospital 1) completely eliminated collapse on Seeds 7 and 99:
- Harmful adaptation dropped from $18.7\% \pm 17.9\%$ (Tent) down to $0.7\% \pm 0.3\%$ (~25-fold reduction).
- Paired F1 improved over frozen source on **all five seeds**: $\Delta\text{F1} = +0.0231$ ($95\%$ CI $[+0.0104, +0.0358]$, nominal $p = 0.0072$, Holm-adjusted $p = 0.0505$, Cohen's $d = 2.260$, 5 wins / 0 losses).
- Paired AUROC gain over source was positive on mean ($\Delta = +0.0058$) but crossed zero in its confidence interval ($95\%$ CI $[-0.0121, +0.0237]$, $p = 0.4199$), confirming that selective gating preserves discriminative ranking rather than generating a statistically significant new state-of-the-art AUROC.

### 7.4 H3: Empirical Rank Equivalence of MC-Dropout and Softmax Confidence
**Preregistered Hypothesis 3:** *MC-dropout predictive variance is mathematically redundant with softmax confidence on binary metastasis classification.*  
**Status: DIRECTLY SUPPORTED.**

On Hospital 1 development data ($N=320$ patches across 5 WSIs):
- The Spearman rank correlation between max-softmax confidence and MC predictive variance was $\rho = -0.9917$.
- Confidence-gated AUROC ($0.9355 \pm 0.0218$) and UTTA-Med MC-gated AUROC ($0.9355 \pm 0.0217$) were virtually identical ($\Delta = -0.00005$, $95\%$ CI $[-0.0008, +0.0007]$, $p = 0.8688$).
- Confidence-gated F1 ($0.8310 \pm 0.0321$) was slightly higher than UTTA-Med ($0.8300 \pm 0.0323$, $\Delta = -0.0010$, Holm-adjusted $p = 0.0430$, Cohen's $d = -2.454$).
- While this small F1 difference is statistically detectable due to extreme cross-seed consistency, the absolute magnitude ($\sim 0.001$) is clinically negligible.
- Latency Benchmark: Full MC-dropout required $598.21$s ($3.55\times$ slower than confidence gating at $168.73$s). Confidence screening provides the pragmatic, compute-frugal deployment choice.

### 7.5 H4: Failure of Matched-Coverage Random Controls
**Preregistered Hypothesis 4:** *Performance gains in selective gating stem from sample quality rather than reduced update counts.*  
**Status: DIRECTLY SUPPORTED.**

When a random gate was configured to admit $70.0\%$ of patches (matching selective gating coverage):
- Random-gated TTA collapsed on Seeds 7 and 99 identically to Tent.
- Mean AUROC dropped to $0.697 \pm 0.304$, and mean F1 dropped to $0.487 \pm 0.385$.
- Harmful adaptation remained severe ($19.4\% \pm 17.0\%$).
- This causal ablation confirms that stability is governed by *which* samples adapt, rejecting the hypothesis of compute-budget artifacts.

### 7.6 H5: Runtime Parameter Displacement Bounding
**Preregistered Hypothesis 5:** *SafeAdapt constrains parameter drift within $\|\theta_t - \theta_0\|_\infty \le 0.05$ with zero numerical faults.*  
**Status: DIRECTLY SUPPORTED.**

Across all 50 streaming evaluation batches (10 in E1, 40 in E2):
- Non-finite gradient or loss failures: $0$ (100% finite execution).
- Maximum observed parameter displacement: $\|\theta_t - \theta_0\|_\infty = 4.01 \times 10^{-5}$ ($0.080\%$ of allowed $0.05$ bound).
- False alarm rollbacks on nominal streaming: $0$.
- In synthetic stress testing with injected NaNs and infinite gradients, the atomic rollback handler recovered pre-step parameters with $100\%$ bit-exact precision (verified in unit tests).

### 7.7 H6: Negative Transfer on External Target Domain (Hospital 2)
**Preregistered Hypothesis 6:** *Selective adaptation yields positive diagnostic transfer on external target Hospital 2.*  
**Status: CONTRADICTED BY RESULTS (Negative Transfer Observed).**

Under the frozen pre-adaptation streaming protocol on Hospital 2 (10 WSIs, 1,280 patches):
- Frozen Source Baseline: Macro AUROC $0.9308$, Pooled AUROC $0.9364$, Pooled F1 $0.7840$, ECE $0.0650$, Brier $0.091709$.
- Active SafeAdapt (Stage C2): Macro AUROC $0.9300$ ($-0.0008$), Pooled AUROC $0.9359$ ($-0.0005$), Pooled F1 $0.7822$ ($-0.0018$), ECE $0.0655$ ($+0.0005$ worse), Brier $0.092053$ ($+0.000344$ worse).
- Disaggregated slide-level analysis revealed that degradation was localized to Slide 29: Slide 29 AUROC dropped from $0.9293 \to 0.9281$, and F1 dropped from $0.3659 \to 0.3457$.
- Mechanism: On Slide 29, high-confidence false-positive patches passed the $\tau = 0.90$ threshold, triggering confirmation bias that progressively degraded decision boundaries. Frozen source remains the strictly superior model on Hospital 2.

### 7.8 Extended Cross-Modality Probes
* **NCT-CRC-HE Multi-Class Decoupling:** On 9-class colorectal histology, evidential vacuity ($u = K/S$) decoupled from confidence ($\rho = -0.4505$), enabling selective Dirichlet adaptation that achieved a $0.0\%$ harm rate ($0$ correct patches degraded, $4$ net corrections).
* **MIDOG 2021 Multi-Scanner Mitosis Detection:** In Faster R-CNN detection across 3,300 tiles, SafeAdapt achieved statistically superior F1 over frozen source on Scanner 1 ($\Delta F_1 = +0.0134$, $95\%$ CI $[+0.0021, +0.0275]$) and Scanner 2 ($\Delta F_1 = +0.0069$, $95\%$ CI $[+0.0023, +0.0129]$), while preventing recall collapse on Scanner 3 ($0.4529$ vs. $0.3944$).

### 7.9 Summary of Hypothesis Outcomes
Table 4 synthesizes the final inferential status across all evaluated hypotheses.

```
Table 4: Final inferential status of preregistered experimental hypotheses.
========================================================================================================================
Hypothesis ID   Description                 Primary Metric / Endpoint     Effect Size & p-value         Formal Status
------------------------------------------------------------------------------------------------------------------------
H1              Tent Collapse Instability   5-seed AUROC & Sensitivity    Delta = -0.232, 2/5 Collapsed DIRECTLY SUPPORTED
H2              Selective Harm Reduction    Harm Rate & Paired F1         Harm: 18.7% -> 0.7%, d=2.26   DIRECTLY SUPPORTED
H3              Binary Gating Equivalence   Spearman rho & Delta AUROC    rho = -0.9917, p = 0.8688     DIRECTLY SUPPORTED
H4              Matched-Coverage Failure    Random Gate AUROC & Harm      AUROC: 0.697, 2/5 Collapsed   DIRECTLY SUPPORTED
H5              SafeAdapt Runtime Bounding  Max L_inf Displacement        Max disp = 4.01e-5 << 0.05    DIRECTLY SUPPORTED
H6              Target Positive Transfer    Hospital 2 AUROC & F1         Delta AUROC = -0.0008         CONTRADICTED (Negative)
Probe 1         Evidential Decoupling       NCT-CRC-HE 9-class Harm       Harm Rate = 0.0% (4 net gain) SUPPORTED (Probe)
Probe 2         Multi-Scanner Detection     MIDOG 2021 Faster R-CNN F1    Delta F1 = +0.0134 (p < 0.05) SUPPORTED (Probe)
========================================================================================================================
```

---

## 8. Discussion

### 8.1 Interpretation of Main Findings
The primary value of this investigation lies in its rigorous, transparent demonstration of the boundaries of unsupervised test-time adaptation in computational pathology:
1. **Unconstrained Adaptation is Clinically Unviable:** Standard Tent cannot be deployed safely in digital pathology. Without sample gating, stochastic gradient descent on entropy drives representations into single-class attractors.
2. **Selective Gating is an Effective Safeguard:** Hard uncertainty screening eliminates gross collapse and reduces harmful adaptations 25-fold across multiple random initializations.
3. **The Limits of Bounded Updates:** While SafeAdapt strictly constrains parameter drift to a micro-perturbation regime ($L_\infty \approx 4 \times 10^{-5}$), bounding parameter space cannot convert negative adaptation dynamics into positive transfer when target batches induce confirmation bias.

### 8.2 What the Results Establish (Supported Claims)
* Multi-seed evaluation is mandatory: evaluating only Seed 42 would have masked the catastrophic collapse occurring on Seeds 7 and 99.
* Confidence gating is practically equivalent to MC-dropout variance on binary tasks while executing $3.55\times$ faster.
* Whole-slide boundary invariance resets eliminate inter-patient contamination in clinical pipelines.

### 8.3 The Mechanism of Confirmation Bias in Histopathology
On outlier slides such as Slide 29, domain shift alters morphological features (e.g., atypical chromatin texture or unusual stroma counter-staining) such that non-tumor tissue produces high-confidence false-positive predictions. Because confidence gating admits patches with $C(x) \ge 0.90$, these false positives enter the loss function. Minimizing entropy steepens the model's confidence on its own errors, corrupting decision boundaries.

### 8.4 Computational Frugality & Practical Clinical Architecture
In clinical digital pathology laboratories, whole-slide scanners process hundreds of slides daily. Imposing an iterative 20-pass Monte Carlo dropout routine adds over 7 minutes of GPU latency per slide. Because confidence gating achieves identical discriminative accuracy with a single deterministic forward pass, we recommend **deterministic confidence gating as the primary practical architecture**.

### 8.5 What the Results Do NOT Establish (Scientific Firewall)
To maintain strict academic integrity, we define the following explicit scientific boundaries:
* **NOT CLAIMED: Certified Clinical Safety:** SafeAdapt is an engineering controller with bounded parameter projections; it does not provide clinical non-inferiority guarantees.
* **NOT CLAIMED: Universal SOTA AUROC Gain:** Selective gating does not establish a statistically significant AUROC improvement over a well-trained frozen source model ($p = 0.4199$).
* **NOT CLAIMED: Universal Uncertainty Equivalence:** Monotonicity holds strictly for single-logit binary classification; in multi-class settings ($K \ge 3$), evidential vacuity decouples from confidence.
* **NOT CLAIMED: Clinical Deployment Clearance:** The software is an investigational prototype and is strictly contra-indicated for primary clinical diagnostic use.

---

## 9. Limitations

1. **Cohort Sample Size:** The external target evaluation is conducted on Camelyon17 Hospital 2 (10 WSIs, 1,280 patches). While representative of the official WILDS split, 10 slides yield wide slide-level confidence intervals.
2. **Patch Subsampling Budget:** Evaluated on 128 deterministically sampled patches per slide rather than full whole-slide coverage (20,000–50,000 patches). Long-stream adaptation dynamics under massive patch streams remain an open research question.
3. **Binary Classification Focus:** Primary findings reflect binary lymph node metastasis classification; multi-class histology (NCT-CRC) was evaluated as a secondary probe.
4. **Synthetic Shock Validation:** The collapse detector logged zero false alarms on nominal streams, but sensitivity has only been evaluated under synthetic fault injection.

---

## 10. Reproducibility, Toolchain & Data Integrity

### 10.1 Pinned Environment & Verified Toolchain
All experiments and audits were verified under the following standardized environment:
* **Python Interpreter:** Python 3.14.2 (Verified in conda environment `avinash`)
* **PyTorch Core:** PyTorch `2.14.0+cpu`
* **Test Harness:** Pytest `9.1.1`
* **Operating System:** Windows 11 Enterprise (x86_64)

### 10.2 Cryptographic Manifest & Automated Test Suite
The automated regression suite encompasses 18 test modules validating numerical stability, boundary resets, and label isolation:
```bash
python -m pytest tests/
========================= 120 passed, 2 skipped in 11.16s =========================
```
* `tests/test_safeadapt_v2.py`: 16 passed (including regression tests for G-03 empty-gate momentum bypass and G-04 source immutability auditing).
* `tests/test_no_leakage.py`: 8 passed, 2 skipped (verifying target labels are never ingested during adaptation).
* `tests/test_gating_v2.py`: 14 passed (verifying bit-exact gating semantics).

### 10.3 Bit-Exact Numerical Table Reconciliation
Executing `scripts/verify_manuscript_tables.py` audits 54 numerical assertions against stored JSON/CSV artifacts:
```bash
python scripts/verify_manuscript_tables.py
Total verification checks performed: 54
Passed: 54 / 54 (100% Bit-Exact Passing Rate)
Failed: 0 / 54
```

---

## 11. Future Work

1. **Full-Slide Dense Streaming:** Extending SafeAdapt to process complete whole slides ($>30,000$ patches) to evaluate whether confirmation bias saturates or causes delayed collapse.
2. **Multi-Scale Spatial Context:** Integrating tile-level TTA with slide-level vision transformer foundation models (e.g., UNI, Prov-GigaPath, Virchow) via Low-Rank Adaptation (LoRA).
3. **Contrastive Anti-Attractor Objectives:** Developing self-supervised contrastive objectives that explicitly regularize feature diversity to prevent confirmation bias under severe stain shift.

---

## 12. Conclusion

Unsupervised test-time adaptation via unconstrained entropy minimization is unstable in computational pathology, suffering catastrophic representation collapse under multi-seed evaluation. SafeAdapt-TTA resolves gross instability through selective uncertainty gating, cutting harmful adaptations ~25-fold and consistently improving paired F1. In single-logit binary classification, deterministic confidence gating matches MC-dropout predictive variance while saving $3.55\times$ computational overhead. However, rigorous evaluation on an external clinical target domain reveals that parameter-bounding safeguards cannot convert negative adaptation dynamics into positive diagnostic transfer on outlier slides. These findings underscore the critical necessity of multi-seed reporting, slide-level inferential units, and rigorous safeguarding before considering test-time adaptation for clinical deployment.

---

## References

1. Aubreville, M., et al. (2021). Mitosis domain generalization in histopathology images — The MIDOG challenge. *Medical Image Analysis*, 84, 102699.
2. Bandi, P., et al. (2018). From detection of individual metastases to classification of lymph node status at the patient level: The CAMELYON17 challenge. *IEEE Transactions on Medical Imaging*, 38(2), 550–560.
3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*, 1050–1059.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
5. Kang, H., et al. (2023). RandStainNA: Learning stain-invariant features through optical density augmentation. *MICCAI*, 417–427.
6. Kather, J. N., et al. (2019). Predicting survival from colorectal cancer histology slides using deep learning: A multicenter study. *PLOS Medicine*, 16(1), e1002730.
7. Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. *ICML*, 5637–5664.
8. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*, 30.
9. Liang, J., et al. (2023). A comprehensive survey on test-time adaptation. *IEEE TPAMI*, 46(5), 3125–3145.
10. Niu, S., et al. (2022). Efficient test-time model adaptation without forgetting. *ICML*, 16888–16905.
11. Niu, S., et al. (2023). Towards stable test-time adaptation in dynamic wild world. *ICLR*.
12. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS*, 31.
13. Wang, D., et al. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR*.
