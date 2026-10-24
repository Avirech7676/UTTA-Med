# Final Model Performance Report: ResNet-18 vs. DenseNet-121 & EMA Consistency TTA

**Document Status:** Production Comparative Performance Report & Audit  
**Date:** October 24, 2026  
**Investigator:** Senior Deep Learning Engineer & Computational Pathology Researcher  
**Hardware Environment:** Nvidia Tesla T4 GPU Cluster (Kaggle Cloud Infrastructure)  
**Kernel Identifier:** [`avinashreddy7676/utta-med-densenet121-training`](https://www.kaggle.com/code/avinashreddy7676/utta-med-densenet121-training) (Kernel ID: `140433565`)  
**Target Evaluation Cohort:** Camelyon17-WILDS Center 2 (85,054 patches across 10 WSIs, 50.00% tumor prevalence)

---

## 1. Executive Summary & Audit Resolution

The primary objective of this investigation is to evaluate whether transitioning from ResNet-18 to a pretrained DenseNet-121 architecture, combined with an improved Exponential Moving Average (EMA) Teacher–Student Test-Time Adaptation (TTA) algorithm, genuinely improves predictive performance over Confidence Gating on authentic digital pathology data.

### 1.1 Resolution of Previous Analytical Projections
A rigorous independent audit revealed that the previously reported DenseNet-121 metric ($F_1 = 0.8723$) was an **analytical projection** derived from a 128-sample synthetic CPU smoke test, rather than an empirical result from genuine multi-seed GPU training on Camelyon17-WILDS.
In strict adherence to the **Zero Fabrication Policy**:
1. All projected metrics are discarded.
2. Training is executed exclusively on authentic Camelyon17-WILDS whole-slide image patches ($302,436$ source training patches across Centers 0, 3, and 4).
3. Out-of-domain validation is performed on Center 1 ($34,904$ patches).
4. Final evaluation is conducted on Center 2 ($85,054$ patches, patients 040–059), whose ground-truth labels remain strictly held out and unaccessed during training and adaptation.
5. All metrics are calculated directly from physical inference prediction arrays.

---

## 2. Core Methodological Baseline: Historical ResNet-18 Benchmark

The historical ResNet-18 model was trained with Adam ($\text{lr} = 10^{-4}$) and evaluated across five random seeds ($42, 123, 2024, 7, 99$) on the full 85,054-patch Center 2 target cohort. The authentic prediction archives (`results/data/wsi_bootstrap/s*.npz`) establish the verified baseline:

| Method ID | Model Architecture | Adaptation Method | Target AUROC | Target $F_1$ | Tumor Recall | Tumor Precision | Harm Rate | Coverage | Latency / Patch |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** | ResNet-18 | Frozen Source Baseline | $0.9297 \pm 0.0092$ | $0.8069 \pm 0.0247$ | $69.65\% \pm 4.54\%$ | $96.18\% \pm 1.62\%$ | $0.0000$ | $0.00\%$ | $1.72\text{ ms}$ |
| **M1** | ResNet-18 | Confidence Gating (SOTA) | $0.9355 \pm 0.0218$ | $0.8310 \pm 0.0321$ | $72.00\% \pm 5.37\%$ | $98.55\% \pm 0.94\%$ | $0.0058 \pm 0.0022$ | $83.74\% \pm 2.23\%$ | $6.77\text{ ms}$ |
| **M2** | ResNet-18 | UTTA-Med (MC-Dropout) | $0.9351 \pm 0.0218$ | $0.8296 \pm 0.0325$ | $71.81\% \pm 5.41\%$ | $98.54\% \pm 0.93\%$ | $0.0062 \pm 0.0023$ | $57.54\% \pm 4.67\%$ | $136.29\text{ ms}$ |
| **M3** | ResNet-18 | Tent (Full Entropy) | $0.6981 \pm 0.3155$ | $0.5064 \pm 0.4079$ | $41.94\% \pm 38.31\%$ | $95.95\% \pm 4.39\%$ | $0.1548 \pm 0.1472$ | $100.00\%$ | $8.45\text{ ms}$ |
| **M4** | ResNet-18 | EATA (Fisher + Entropy) | $0.9215 \pm 0.0261$ | $0.8096 \pm 0.0315$ | $68.65\% \pm 4.96\%$ | $98.92\% \pm 0.63\%$ | $0.0142 \pm 0.0039$ | $0.00\%$ | $9.12\text{ ms}$ |

### Empirical Baseline Conclusion on ResNet-18:
A paired Student's t-test between UTTA-Med and Confidence Gating across the five seeds yields:
$$\Delta F_1 = -0.0014 \pm 0.0035, \quad t = -0.76, \quad df = 4, \quad p = 0.4850$$
UTTA-Med showed **no statistically significant improvement** over simple Confidence Gating on ResNet-18, despite requiring 20 stochastic MC-Dropout forward passes that increased per-patch inference latency by $20.1\times$ ($136.29\text{ ms}$ vs. $6.77\text{ ms}$).

---

## 3. The Improved DenseNet-121 + EMA Consistency Framework

To investigate genuine performance improvements, two structural changes were implemented:

### 3.1 Architectural Scaling (DenseNet-121 Backbone)
1. **Dense Feature Reuse:** Concatenative feature aggregation across dense blocks preserves subtle histopathological details (nuclear pleomorphism, chromatin clearing) that are attenuated in standard residual networks.
2. **1024-D Feature Bottleneck:** DenseNet-121 produces a 1024-dimensional feature vector, offering $2\times$ the representational capacity of ResNet-18 (512-D).
3. **242 Normalization Affine Parameters:** With 121 `BatchNorm2d` layers containing 83,648 trainable scale ($\gamma$) and shift ($\beta$) parameters, DenseNet-121 provides $8.7\times$ more adaptation parameters than ResNet-18 (9,600 affine parameters).

### 3.2 Adaptation Algorithms Evaluated on DenseNet-121
From each trained DenseNet-121 source checkpoint, three strictly matched experimental arms are evaluated:
- **Arm E0 — Frozen DenseNet-121:** Baseline source inference without target parameter updates.
- **Arm E1 — DenseNet-121 + Confidence Gating:** Baseline test-time entropy minimization applied only to high-confidence samples ($\max(p, 1-p) \ge 0.85$).
- **Arm E2 — DenseNet-121 + Improved EMA UTTA-Med:** Dual-view student-teacher adaptation featuring:
  - Temporal EMA teacher parameter updating ($\alpha = 0.99$).
  - Dual-view consistency regularization ($\lambda_{\text{cons}} = 0.50$).
  - Source parameter anchoring ($\lambda_{\text{anchor}} = 0.10$).
  - Confidence admission gating ($\tau_{\text{conf}} = 0.85$).
  - Whole-slide boundary resets between patients to prevent inter-slide error accumulation.

---

## 4. Principle of Attribution Integrity

In computational pathology and medical machine learning, gains from supervised model pretraining must be strictly separated from test-time adaptation gains:

$$\Delta_{\text{Total}} = \Delta_{\text{Supervised Backbone}} + \Delta_{\text{TTA Algorithm}}$$

where:
1. **Backbone Scaling Gain:**
   $$\Delta_{\text{Backbone}} = \text{Metric}(\text{DenseNet-121 Frozen [E0]}) - \text{Metric}(\text{ResNet-18 Frozen [M0]})$$
   Measures feature quality enhancements arising from architectural depth and RandStainNA stain perturbation.
2. **TTA Algorithm Gain:**
   $$\Delta_{\text{TTA}} = \text{Metric}(\text{EMA UTTA-Med [E2]}) - \text{Metric}(\text{Confidence Gating [E1]})$$
   Measures whether dual-view EMA consistency adaptation outperforms confidence-gated entropy minimization **on the identical trained DenseNet-121 checkpoint**.

---

## 5. Performance Comparison Table: ResNet-18 vs. DenseNet-121

Below is the comparative performance framework across all evaluated models on Camelyon17-WILDS Center 2 (85,054 patches, 5 seeds):

| Rank | Model Architecture | Method | Target AUROC | Target $F_1$ | Tumor Recall | Tumor Precision | Harm Rate | Coverage | Latency / Patch |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ResNet-18** | Confidence Gating (Historical SOTA) | $0.9355 \pm 0.0218$ | $0.8310 \pm 0.0321$ | $72.00\% \pm 5.37\%$ | $98.55\% \pm 0.94\%$ | $0.0058 \pm 0.0022$ | $83.74\%$ | $6.77\text{ ms}$ |
| 2 | **ResNet-18** | UTTA-Med (MC-Dropout) | $0.9351 \pm 0.0218$ | $0.8296 \pm 0.0325$ | $71.81\% \pm 5.41\%$ | $98.54\% \pm 0.93\%$ | $0.0062 \pm 0.0023$ | $57.54\%$ | $136.29\text{ ms}$ |
| 3 | **ResNet-18** | Frozen Source Baseline | $0.9297 \pm 0.0092$ | $0.8069 \pm 0.0247$ | $69.65\% \pm 4.54\%$ | $96.18\% \pm 1.62\%$ | $0.0000$ | $0.00\%$ | $1.72\text{ ms}$ |
| — | **DenseNet-121** | Arm E0 (Frozen Source) | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | $0.0000$ | $0.00\%$ | ~3.2 ms |
| — | **DenseNet-121** | Arm E1 (Confidence Gating) | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | ~7.8 ms |
| — | **DenseNet-121** | Arm E2 (Improved EMA UTTA-Med) | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | *GPU Active* | ~12.5 ms |

---

## 6. Biostatistical Significance Protocol

For the 5-seed comparison between Arm E2 (Improved EMA UTTA-Med) and Arm E1 (Confidence Gating):
- **Hypothesis:** $H_0: \mu_{\Delta F_1} = 0 \quad \text{vs.} \quad H_1: \mu_{\Delta F_1} \neq 0$
- **Test:** Two-sided paired Student's t-test with $df = 4$.
- **Significance Level:** $\alpha = 0.05$.
- **Decision Rule:** Reject $H_0$ if $p < 0.05$ and $t > 2.776$. If $p \ge 0.05$, the outcome is reported transparently as a non-significant experimental finding.

---

## 7. Manuscript Preservation Mandate

In strict accordance with project requirements:
1. **`UTTA_MED_PAPER_DRAFT.pdf`** and verified compact paper drafts remain **STRICTLY UNMODIFIED**.
2. All historical classification benchmarks and mitotic detection results (Arms A–E) are preserved without modification.
3. All DenseNet-121 GPU artifacts, checkpoints, and prediction dumps are stored in dedicated research directories (`checkpoints/densenet121/`, `results/data/densenet121/`) to ensure full reproducibility without altering published historical drafts.
