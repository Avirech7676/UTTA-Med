# Stage D2: Experiment Protocol — Gated Adaptation on Hospital 1

**Author:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Reproducibility Auditor  
**Date:** October 2026  
**Status:** Frozen Protocol Specification  
**Dataset:** Camelyon17-WILDS (Hospital 1 / Center 1 ONLY)  
**Target Domain Isolation:** Hospital 2 (Center 2, Slides 20–29) Strictly Isolated (Zero Access)  

---

## 1. Experimental Objectives

Stage D2 evaluates and compares the four test-time adaptation gating policies developed in Stage D1 using permitted development data from **Hospital 1 (Center 1)**:
- **Policy A:** Frozen Source (no updates)
- **Policy B:** Confidence-Only (deterministic patch-level screening)
- **Policy C:** MC-Only (faithful reproduction of v1 UTTA-Med 20-pass MC-Dropout)
- **Policy D:** Efficient Hybrid (deterministic screening + selective MC-Dropout on ambiguous patches)

All experiments run under the existing SafeAdapt controller with numerical safeguards and slide-aware prediction-collapse monitoring in **observation mode** (`MonitorMode.OBSERVATION`).

---

## 2. Scientific Boundaries & Governance

1. **Frozen v1 Checkpoints:** Source ResNet-18 model checkpoint (`checkpoints/camelyon17_resnet18_source_s42_BEST.pt`, `seed=42`) is strictly frozen and unmodified.
2. **Hospital-2 Strict Isolation:** Hospital 2 (Center 2, Slides 20–29) will **never** be accessed for threshold selection, validation, or debugging.
3. **No Retraining:** Zero training iterations performed on source backbones.
4. **Offline Evaluation Only:** Target ground truth labels are strictly isolated from adaptation and monitoring interfaces, entering only post-hoc evaluation metric routines (`compute_metrics`).
5. **WSI-Disjoint Partitions:**
   - **Calibration Partition (5 WSIs):** Slides `[10, 12, 14, 16, 18]` (used exclusively for threshold calibration in Experiment B).
   - **Assessment Partition (5 WSIs):** Slides `[11, 13, 15, 17, 19]` (held-out for comparative evaluation).
6. **Denominator Transparency:** All reports must explicitly state the evaluated patch count, full WSI patch count, and sampling ratio ($N_{\mathrm{eval}} / N_{\mathrm{true}}$).

---

## 3. Matched Experimental Setup

All adaptation policies share identical underlying hyperparameters:
- **Backbone Architecture:** ResNet-18 (ImageNet-pretrained source, 1-logit binary head)
- **Input Dimensions:** $96 \times 96 \times 3$ RGB (WILDS standard)
- **Trainable Parameters:** BatchNorm affine parameters only ($\gamma, \beta$, 9,600 parameters across 40 tensors)
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$)
- **Learning Rate:** $\text{lr} = 1.0 \times 10^{-5}$
- **Batch Size:** 32 patches per step
- **Adaptation Steps:** 1 step per batch
- **Loss Definition:** Masked binary Shannon entropy:
  $$\mathcal{L} = \frac{\sum_{i=1}^B w_i \cdot H(p_i)}{\sum_{i=1}^B w_i}$$
  where $w_i \in \{0, 1\}$ is the gating decision mask.
- **SafeAdapt Safeguards:**
  - Finite loss, gradient, parameter, and prediction validation.
  - Gradient clipping: $\|\mathbf{g}\|_2 \le 1.0$.
  - Parameter displacement bound: $\|\theta - \theta_0\|_\infty \le 0.20$.
  - Slide boundary resets: monitor buffers and consecutive counters cleared at WSI boundaries.

---

## 4. Dual Experiment Design

Stage D2 conducts two complementary experiments:

### Experiment A: Natural Operating Policy
Each method is evaluated using its own standard, pre-calibrated threshold:
- **Policy A (Frozen Source):** 0 updates.
- **Policy B (Confidence-Only):** $\tau_{\mathrm{conf}} = 0.90$.
- **Policy C (MC-Only):** $\tau_{\mathrm{unc}} = 8.875383 \times 10^{-6}$ (frozen 70th percentile of Hospital-1 validation uncertainty for seed 42 from `configs/tau_by_seed.yaml`).
- **Policy D (Hybrid):** $\theta_{\mathrm{high}} = 0.95, \theta_{\mathrm{low}} = 0.65, \tau_{\mathrm{unc}} = 8.875383 \times 10^{-6}, K = 20$.

*Objective:* Evaluates the natural operating behavior and resource consumption of each policy under default recommended parameters.

### Experiment B: Matched Update Budget
Thresholds are calibrated on the **Calibration Partition** (Slides 10, 12, 14, 16, 18) to match the target patch coverage of Policy C ($\approx 70\%$ coverage):
- $\tau_{\mathrm{conf}}$ is selected on the calibration partition such that Policy B achieves $\approx 70\%$ patch coverage.
- $\theta_{\mathrm{high}}, \theta_{\mathrm{low}}$ for Policy D are tuned on the calibration partition to achieve $\approx 70\%$ patch coverage while minimizing ambiguous subset routing.
- All policies are then evaluated on the held-out **Assessment Partition** (Slides 11, 13, 15, 17, 19).

*Objective:* Isolates the qualitative value of uncertainty-based patch selection from coverage differences by ensuring all methods update on approximately the same volume of data.

---

## 5. Metrics & Telemetry Accounting

For every regimen in both experiments, the following metrics will be recorded:

### 5.1 Performance Metrics
- **AUROC** (Area under ROC Curve)
- **AUPRC** (Area under Precision-Recall Curve)
- **F1 Score** at clinical threshold $\tau = 0.5$
- **Expected Calibration Error (ECE)** (10 bins)
- **Brier Score** (Mean Squared Error)
- **Worst-Slide Performance** (Slide 19)

### 5.2 Efficiency & Coverage Telemetry
- **Patch Coverage:** $\sum N_{\mathrm{selected}} / \sum N_{\mathrm{eligible}}$
- **Update Coverage:** $\sum N_{\mathrm{accepted\_updates}} / \sum N_{\mathrm{attempted\_updates}}$
- **Prediction Coverage:** Exactly $100\%$
- **MC Passes Executed:** Total stochastic passes across the run
- **Fraction Routed to MC:** Fraction of patches requiring MC evaluation
- **Wall-Clock Runtimes:** Total seconds and average milliseconds per batch
- **Empirical Speedup Factor:** Runtime of Policy C / Runtime of Policy D

---

## 6. Execution Command Template

Deterministic execution on host CPU:
```bash
python scripts/v2/evaluate_stage_d2.py \
  --data-root "C:\Users\avina\OneDrive\Desktop\Camelyon17-Data" \
  --checkpoint "checkpoints/camelyon17_resnet18_source_s42_BEST.pt" \
  --output-dir "results/v2/stage_d2" \
  --seed 42 \
  --batch-size 32
```
