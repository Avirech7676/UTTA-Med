# Final Model Performance Comparison: ResNet-18 vs. DenseNet-121

**Document Status:** Production Comparative Performance Audit  
**Date:** October 23, 2026  
**Target Evaluation Cohort:** Camelyon17-WILDS Center 2 (85,054 patches, 10 WSIs, 50.00% tumor prevalence)  
**Evaluation Protocol:** Strictly Held-Out Target Evaluation across 5 Random Seeds (42, 123, 2024, 7, 99)

---

## 1. Experimental Overview & Attribution Integrity

A central requirement of rigorous deep learning research in medical imaging is **attribution integrity**:
> **Principle of Attribution Separation:** Performance improvements resulting from a stronger supervised backbone (e.g. transitioning from ResNet-18 to DenseNet-121) must never be conflated with or attributed to the test-time adaptation (TTA) algorithm.

To satisfy this requirement:
1. All adaptation algorithms (Confidence Gating and EMA Consistency UTTA-Med) are evaluated **from the identical source checkpoint** for each seed.
2. The delta $\Delta_{\text{TTA}} = \text{Metric}(\text{TTA}) - \text{Metric}(\text{Confidence Gating})$ is computed strictly within the same trained model.
3. The delta $\Delta_{\text{Backbone}} = \text{Metric}(\text{DenseNet-121 Frozen}) - \text{Metric}(\text{ResNet-18 Frozen})$ measures representation gains from supervised architecture scaling.

---

## 2. Historical ResNet-18 Verified Baseline Performance

The historical ResNet-18 models were trained on source Centers 0, 3, 4 and evaluated on the full 85,054-patch Center 2 cohort across all 5 seeds. The authentic prediction dumps (`results/data/wsi_bootstrap/s*.npz`) establish the following ground-truth baseline:

| Architecture | Method | Target AUROC | Target $F_1$ | Tumor Recall | Tumor Precision | Harm Rate | Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | Frozen Source Baseline (C0) | $0.9297 \pm 0.0092$ | $0.8069 \pm 0.0247$ | $69.65\% \pm 4.54\%$ | $96.18\% \pm 1.62\%$ | $0.0000$ | $0.00\%$ |
| **ResNet-18** | Confidence Gating (SOTA) | $0.9355 \pm 0.0218$ | $0.8310 \pm 0.0321$ | $72.00\% \pm 5.37\%$ | $98.55\% \pm 0.94\%$ | $0.0058 \pm 0.0022$ | $83.74\% \pm 2.23\%$ |
| **ResNet-18** | UTTA-Med (MC-Dropout) | $0.9351 \pm 0.0218$ | $0.8296 \pm 0.0325$ | $71.81\% \pm 5.41\%$ | $98.54\% \pm 0.93\%$ | $0.0062 \pm 0.0023$ | $57.54\% \pm 4.67\%$ |
| **ResNet-18** | Tent (Unconstrained Entropy) | $0.6981 \pm 0.3155$ | $0.5064 \pm 0.4079$ | $41.94\% \pm 38.31\%$ | $95.95\% \pm 4.39\%$ | $0.1548 \pm 0.1472$ | $100.00\%$ |
| **ResNet-18** | EATA (Fisher + Entropy) | $0.9215 \pm 0.0261$ | $0.8096 \pm 0.0315$ | $68.65\% \pm 4.96\%$ | $98.92\% \pm 0.63\%$ | $0.0142 \pm 0.0039$ | $0.00\%$ |

### Critical Finding on ResNet-18:
Paired statistical comparison between UTTA-Med and Confidence Gating on ResNet-18 yields:
$$\Delta F_1 = -0.0014 \pm 0.0035, \quad t = -0.76, \quad df = 4, \quad p = 0.4850$$
UTTA-Med was **statistically indistinguishable** from Confidence Gating while incurring an 8.8× latency overhead due to 20 stochastic MC-Dropout passes.

---

## 3. New DenseNet-121 Architectural Framework

### 3.1 Architectural Advantages
DenseNet-121 introduces structural mechanisms that directly combat domain shift:
1. **Dense Feature Reuse:** Each layer receives concatenated feature maps from all preceding layers, preventing the attenuation of early morphological signals (nuclear pleomorphism, chromatin distribution).
2. **1024-D Bottleneck Representation:** Generates rich, linearly separable feature embeddings compared to the 512-D bottleneck of ResNet-18.
3. **242 Normalization Affine Parameters:** With 121 normalization layers containing 83,648 trainable scale ($\gamma$) and shift ($\beta$) parameters, DenseNet-121 provides 8.7× greater adaptation capacity than ResNet-18 (9,600 affine parameters).

### 3.2 Adaptation Arms Evaluated
- **Arm E0 (Frozen Source DenseNet-121):** Evaluates whether supervised pretraining with RandStainNA optical density augmentation alone elevates out-of-domain feature robustness.
- **Arm E1 (DenseNet-121 + Confidence Gating):** Provides the true competitive adaptation baseline on the new backbone.
- **Arm E2 (DenseNet-121 + EMA Consistency UTTA-Med):** Introduces temporal EMA teacher guidance ($\alpha = 0.99$), dual-view RandStainNA consistency ($\lambda = 0.50$), source parameter anchoring ($\lambda = 0.10$), and slide-boundary resets.

---

## 4. Empirical Evaluation Protocol

All models are evaluated on the identical 85,054 target patches with:
1. **Decision Threshold:** Fixed at $\tau = 0.50$ (calibrated on out-of-domain Center 1 validation data).
2. **Streaming Order:** Sequential patch evaluation by slide (Slides 20 through 29).
3. **Harm Rate Definition:** Proportion of target patches that were correctly classified by the frozen source model but erroneously flipped to incorrect predictions following test-time adaptation:
   $$\text{Harm Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{frozen}}^{(i)} = y^{(i)} \;\land\; \hat{y}_{\text{adapted}}^{(i)} \neq y^{(i)})}{N_{\text{total}}}$$
4. **Coverage Definition:** Proportion of target samples admitted for parameter updates based on admission threshold $\max(p, 1-p) \ge 0.85$.
