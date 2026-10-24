# ORIGINAL RESEARCH REQUIREMENTS SPECIFICATION

**Authoritative Manuscript Reference:**  
*Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
[`UTTA_MED_FINAL_COMPACT_PAPER.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.tex) | [`paper/ieee/20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/paper/ieee/20_FINAL_MANUSCRIPT.tex) | [`UTTA_MED_PAPER_DRAFT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_PAPER_DRAFT.pdf)

**Audit Date:** October 2026 | **Branch:** `research-project-2`  
**Standard:** Complete Reconstruction from Authoritative Manuscript

---

## 1. RESEARCH IDENTITY & CORE MISSION

### 1.1 Official Title (Manuscript Title Block)
- **Requirement:** The approved project title is strictly:
  $$\text{\textbf{Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift}}$$
- **Scope:** The research investigates reliability, failure containment, and selective test-time adaptation under real multicenter and multi-scanner domain shift in computational pathology.

### 1.2 Principal Research Questions (Manuscript Section 1: Introduction)
- **Primary Operational Hypothesis:** Can selective uncertainty gating and fail-closed monitoring contain TTA failure modes without inducing negative adaptation?
- **Core Investigation:**
  1. Does unconstrained entropy minimization (Tent) suffer catastrophic collapse under real histopathologic domain shifts?
  2. Does selective admission gating prevent collapse and reduce harmful adaptation rates?
  3. Does Monte Carlo Dropout predictive variance provide measurable diagnostic accuracy gains over scalar confidence gating in single-logit binary classification?
  4. Can fail-closed circuit breakers and detection-silence monitors contain detector divergence on multi-scanner target streams without clinical degradation?

---

## 2. TRACK A: MULTICENTER METASTASIS CLASSIFICATION (Camelyon17-WILDS)

### 2.1 Dataset Partitioning & Leakage Quarantine (Manuscript Section 4.1)
- **Dataset:** Camelyon17-WILDS $96 \times 96$ px lymph node histopathology patches, binary classification (tumor vs. normal central $32 \times 32$ px region).
- **Source Training Centers:** Centers $\{0, 3, 4\}$ ($n = 302{,}436$ patches across 30 WSIs).
- **In-Domain Validation:** Centers $\{0, 3, 4\}$ ($n = 33{,}560$ patches).
- **Held-Out OOD Validation:** Center 1 (Canisius-Wilhelmina Hospital, $n = 34{,}904$ patches across 10 WSIs). Hyperparameters (admission threshold $\tau$, learning rate) are calibrated strictly on Center 1.
- **Target Test Benchmark:** Center 2 (UMC Utrecht, $n = 85{,}054$ patches across 10 WSIs from exactly 9 unique patients: $\{40, 41, 42, 44, 45, 46, 48, 51, 52\}$). Patient 46 contributes Slides 25 and 26.
- **Leakage Policy:** Target Center 2 labels are strictly quarantined during adaptation and hyperparameter tuning.

### 2.2 Classification Architecture (Manuscript Section 3.1 & Model Module)
- **Backbone:** ResNet-18 ($11{,}449{,}301$ total parameters), single scalar logit output $z(x) \in \mathbb{R}$, sigmoid predictive probability $p(x) = \sigma(z(x))$.
- **Classification Head:** Linear($512 \to 512$) $\to$ ReLU $\to$ Dropout($p=0.5$) $\to$ Linear($512 \to 1$).
- **Source Checkpoints:** 5 certified checkpoints representing independent optimization runs: Seeds $\{42, 123, 2024, 7, 99\}$.

### 2.3 Trainable Parameter Scope & Running Statistics (Manuscript Section 3.1)
- **Trainable Parameters:** Strictly restricted to BatchNorm2d affine parameters ($\gamma, \beta$). ResNet-18 contains 20 BatchNorm2d layers $= 40$ affine parameters.
- **Frozen Parameters:** All convolutional kernels, linear weights, and backbone parameters remain strictly frozen (`requires_grad = False`).
- **Running Statistics Policy:** `track_running_stats = True` with running mean and running variance buffers strictly frozen (`eval()` mode). Target mini-batch statistics are never accumulated into model buffers (`bn_freeze_stats`).

### 2.4 Uncertainty Estimation: Monte Carlo Dropout (Manuscript Section 3.2)
- **Dropout Activation:** Dropout ($p=0.5$) in the classification head remains active during uncertainty estimation while all other layers remain in evaluation mode.
- **Stochastic Passes:** $N = 20$ stochastic forward passes per patch:
  $$\bar{p}(x) = \frac{1}{N}\sum_{j=1}^N p_j(x), \quad U(x) = \frac{1}{N}\sum_{j=1}^N \bigl(p_j(x) - \bar{p}(x)\bigr)^2$$
- **Feature-Caching Equivalence:** Feature activations prior to the head may be cached to accelerate inference, provided output equivalence to full stochastic passes is numerically preserved.

### 2.5 Selective Admission Gating (Manuscript Section 3.2)
- **Binary Hard Gate:**
  $$w(x) = \mathbf{1}\bigl[U(x) < \tau\bigr]$$
  where $\tau$ is calibrated to the 70th percentile of predictive variance $U$ on held-out validation Center 1.
- **Confidence Gate Baseline:** $w_{\text{conf}}(x) = \mathbf{1}\bigl[\max(p, 1-p) \ge \tau_{\text{conf}}\bigr]$, calibrated to matched admission coverage on validation Center 1.
- **Batch Skipping:** If all patches in an incoming mini-batch are rejected ($\sum_i w_i = 0$), the optimizer update step is completely skipped.

### 2.6 Test-Time Adaptation Objective (Manuscript Section 3.2)
- **Loss Function:** Binary Shannon entropy minimization computed exclusively over admitted patches:
  $$\mathcal{L} = \frac{\sum_i w_i H(p_i)}{\sum_i w_i}, \quad H(p) = -p \log p - (1-p) \log(1-p)$$
- **Optimizer:** Adam optimizer with learning rate $\eta = 10^{-5}$, single inner gradient step ($k=1$) per batch.

### 2.7 Fail-Closed Safeguards & Slide Boundary Resets (Manuscript Section 3.1 & 3.3)
- **Parameter Displacement Sphere:** $L_\infty$ distance from source parameters is bounded:
  $$\|\theta_t - \theta_0\|_\infty \le 0.05$$
  Violations trigger immediate rejection of the update and rollback to $\theta_0$.
- **Numerical Anomaly Protection:** Updates producing NaN gradients or non-finite losses are rejected and optimizer state is flushed.
- **Episodic Slide Resets:** At every whole-slide image boundary, model parameters are reset to exact source weights ($\Delta\theta = 0.000000$) and optimizer state is reinitialized, preventing inter-slide error propagation.

---

## 3. TRACK B: DENSE MITOTIC FIGURE DETECTION (MIDOG 2021)

### 3.1 Dataset & Multi-Scanner Shift Protocol (Manuscript Section 4.2)
- **Dataset:** MIDOG 2021 human breast carcinoma mitotic figure detection.
- **Source Training Scanners:** Scanners 1 and 2 (Hamamatsu NanoZoomer 2.0-HT and S360, Cases 001–100, $0.23\,\mu\text{m/pixel}$).
- **Target Test Scanner:** Scanner 3 (Aperio ScanScope CS2, Cases 101–150, 50 held-out cases, $0.25\,\mu\text{m/pixel}$).
- **Spatial Sampling Frame:** 4x4 spatially stratified grid across standardized 10-HPF tumor regions ($1.96\,\text{mm}^2$ per case), 22 crops per case $= 1{,}100$ total evaluation patches ($512 \times 512$ px), capturing 118 verified ground-truth mitoses across 101 positive windows.

### 3.2 Detection Architecture (Manuscript Section 3.1)
- **Detector:** Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN) backbone.
- **Trainable Parameters:** 53 normalization layers in backbone promoted to trainable affine scale $\gamma$ and shift $\beta$ (106 total parameters). Running mean and variance buffers remain frozen.
- **Optimizer:** SGD ($\mathrm{lr} = 2.5 \times 10^{-4}$, momentum $= 0.9$).

### 3.3 Original Detection Experimental Arms A–E (Manuscript Section 3.3)
- **Arm A (Frozen Source Baseline):** Zero adaptation; evaluates target patches with frozen source parameters.
- **Arm B (Conventional TTA):** Adapts 106 BatchNorm affine parameters via self-selected Region of Interest (RoI) proposal entropy at operating threshold $\tau = 0.20$ without safety constraints.
- **Arm C (Bounded TTA):** Proposal entropy minimization constrained to an $L_\infty$ ball centered at source parameters ($\|\theta_{\text{adapt}} - \theta_{\text{source}}\|_\infty \le 0.05$).
- **Arm D (Monitored TTA):** Bounded adaptation with active safety monitoring. Alarms execute atomic parameter and optimizer rollback to last stable checkpoint ($\Delta W = 0.00000000$), permitting subsequent updates.
- **Arm E (SafeAdapt):** Source-anchored admission gating combined with dual-stream recovery: transient alarms trigger atomic rollback; persistent failure ($\ge 3$ consecutive alarms) triggers latching global fallback to source parameters for the remainder of the patch stream.

### 3.4 Detection Safety Monitoring (Manuscript Section 3.3)
- **Detection Silence Monitor:** Trips an alarm if adapted detector predicts 0 detections across $W_{\text{silence}} = 30$ consecutive patches.
- **Absolute Proliferation Monitor:** Trips if $n_{\text{pred}} > 4$ mitoses in a single $512 \times 512$ tile ($>3\times$ biological density in 10-HPF regions).
- **Relative Proliferation Monitor:** Trips if $n_{\text{source}} > 0$ and $n_{\text{pred}} > 3.0 \times n_{\text{source}}$.

---

## 4. SCIENTIFIC FINDINGS & NEGATIVE RESULT MANDATE

### 4.1 Authentic Classification Findings (Manuscript Section 5)
- Unconstrained Tent collapses on 2 of 5 seeds (Seeds 7 and 99, $F_1 \approx 0.05$, AUROC $\approx 0.34$).
- Selective gating eliminates collapse, reduces harmful adaptation rate from $0.187 \pm 0.179$ to $0.007 \pm 0.003$ ($>25\times$), and improves $F_1$ across all five seeds ($+0.023$, $95\%$ CI $[0.010, 0.036]$).
- Monte Carlo Dropout variance (UTTA-Med) achieves similar observed accuracy to scalar softmax confidence gating ($p = 0.869$), while confidence gating is $17.7\times$ faster.

### 4.2 Authentic Detection Findings (Manuscript Section 6)
- SafeAdapt achieves identical mean $F_1$ to frozen source ($0.4774$, observed $\Delta F_1 = 0.0000$, all $10{,}000$ paired bootstrap replicates equal $0.0000$).
- Early detection-silence alarms trigger permanent source fallback on all three seeds (Seed 42 at patch 102; Seed 7 at patch 80; Seed 123 at patch 58), preserving source performance rather than achieving positive target adaptation.

### 4.3 Mandatory Scientific Reporting Integrity
- Failure containment is not positive adaptation gain or clinical validation.
- All negative results must be preserved with complete integrity without fabricating replacement architectures.
