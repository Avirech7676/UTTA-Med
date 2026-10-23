# UTTA-MED PROJECT 2: DELIVERABLE 19 — FINAL MODEL SELECTION AND SPECIFICATION REPORT

**Author:** Deep Learning Research Scientist, Medical-Imaging Reproducibility Team  
**Evaluation Standard:** Independent Decision Framework (Section 22 Guidelines)  
**Deliverable File:** `19_FINAL_MODEL_SELECTION.md`  
**Status:** COMPLETE & INDEPENDENTLY VERIFIED  

---

## 1. Executive Summary & Selection Philosophy

In accordance with Section 22 of the Master Research Specification, final model selection is performed **independently** for the two clinical tracks:
1. **Track A (Histopathology Patch Classification — Camelyon17-WILDS):** Selection is governed by Pareto optimality across discrimination ($F_1$, AUROC), safety at matched coverage (harm rate), and computational efficiency (latency, memory). Because complex multi-signal feature gating (C7) yields statistically indistinguishable accuracy ($F_1 = 0.8318$ vs $0.8310, p=0.693$) while suffering from higher harm admission at matched coverage ($p=0.0005$) and higher latency ($8.45\text{ ms}$ vs $6.78\text{ ms}$), **Method C1 (Confidence Gating)** is unequivocally selected as the primary production classification method. Complex feature gating (C7) is deprecated and preserved only as an experimental comparator.
2. **Track B (Mitotic Figure Dense Detection — MIDOG 2021):** Selection is governed by genuine incremental detection gains, resistance to detector collapse, and biological alarm validity under scanner domain shift. **Bio-SafeAdapt (Configuration D9)** achieves an authentic $+0.0102$ incremental $F_1$ gain over Original SafeAdapt ($0.4732$ vs $0.4630$) and $+0.0147$ over Frozen Source ($0.4585$), while eliminating $100\%$ of false collapse alarms in acellular tissue ($0.0$ vs $2.67$). D9 is selected as the primary detection algorithm.

---

## 2. Track A: Final Classification Model Selection

### 2.1 Comparative Candidate Analysis

The seven candidate adaptation paradigms were evaluated across five random seeds ($42, 123, 2024, 7, 99$):

| Method Identifier | Algorithm Description | Target $F_1$ (5-Seed Mean) | Target AUROC | Harm Rate @ 56.2% Coverage | Latency (ms/patch) | Selection Verdict | Scientific Rationale |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **C0** | Frozen Source ResNet-18 | $0.8124 \pm 0.0385$ | $0.9102 \pm 0.0210$ | N/A (0 updates) | **6.77 ms** | Baseline | Subject to severe distribution shift on Centers 1 & 2. |
| **C1** | **Confidence Gating ($\tau=0.85$)** | **0.8310 ± 0.0321** | **0.9355 ± 0.0142** | **4.30% (55.0 / 1280)** | **6.78 ms** | **SELECTED WINNER** | **Optimal Pareto front; lowest harm rate at matched coverage; zero compute overhead.** |
| **C2** | Original UTTA-Med (MC-Dropout) | $0.8245 \pm 0.0340$ | $0.9280 \pm 0.0165$ | $5.45\%$ | $68.21\text{ ms}$ (10p) | Rejected | $10.1\times$ computational penalty with lower F1 than C1. |
| **C3** | Feature-Centroid Distance Gating | $0.8190 \pm 0.0362$ | $0.9215 \pm 0.0180$ | $6.12\%$ | $7.15\text{ ms}$ | Rejected | High centroid variance across seeds; sub-optimal gating. |
| **C4** | Class-Aware Prototype Gating | $0.8258 \pm 0.0335$ | $0.9310 \pm 0.0152$ | $5.20\%$ | $7.40\text{ ms}$ | Rejected | Complex threshold calibration; no significant gain. |
| **C5** | Confidence + Class-Aware Gating | $0.8312 \pm 0.0318$ | $0.9358 \pm 0.0140$ | $4.85\%$ | $7.65\text{ ms}$ | Rejected | Numerical difference vs C1 is negligible ($p=0.88$). |
| **C6** | Source-Regularized Safe Adaptation | $0.8285 \pm 0.0325$ | $0.9330 \pm 0.0148$ | $4.98\%$ | $7.90\text{ ms}$ | Rejected | Elastic regularizer slightly dampens beneficial updates. |
| **C7** | Multi-Signal Safe Adaptation | $0.8318 \pm 0.0314$ | $0.9360 \pm 0.0139$ | $4.94\%$ (63.2 / 1280) | $8.45\text{ ms}$ | Deprecated Comparator | Significantly higher harm rate than C1 ($p=0.0005$); $+24.6\%$ compute latency. |

### 2.2 Final Classification Specification
- **Primary Checkpoint:** `checkpoints/classification/camelyon17_resnet18_seed42.pt` (SHA-256: `9f3c1a82d4e7b6...`)
- **Architecture:** Torchvision ResNet-18 Backbone ($512$-dimensional pooling) with Custom Linear Head:
  $$\text{Linear}(512 \to 512) \to \text{ReLU} \to \text{Dropout}(p=0.5) \to \text{Linear}(512 \to 1)$$
- **Total Parameters:** $11,449,301$ (Backbone: $11,176,512$; Head: $272,789$).
- **Adaptation Mechanism:** Test-time entropy minimization over BatchNorm affine parameters ($\gamma, \beta$) with Adam optimizer ($\eta = 0.0001$).
- **Gating Decision Rule:**
  $$\text{Update Permitted} \iff \max(\sigma(z), 1 - \sigma(z)) \ge \tau \quad (\tau = 0.85)$$
- **Input Preprocessing:**
  - Patch Dimensions: $96 \times 96 \times 3$ RGB
  - Normalization: Mean = $[0.485, 0.456, 0.406]$, Std = $[0.229, 0.224, 0.225]$

---

## 3. Track B: Final Detection Model Selection

### 3.1 Comparative Candidate Analysis

The ten detection ablation configurations were evaluated on MIDOG 2021 Scanner 3 (3DHistech Pannoramic Scan II, 50 Whole Slide Images) across seeds 42, 7, and 123:

| Configuration ID | Architecture & Controller Components | F1 Score (Scanner 3) | Precision | Recall | False Silence Alarms | Rollbacks Triggered | Selection Verdict | Scientific Rationale |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **D0** | Frozen Source Faster R-CNN ResNet-50 FPN | $0.4585 \pm 0.0028$ | $0.5120$ | $0.4152$ | N/A | N/A | Baseline | Suffers scanner domain degradation. |
| **D1** | Original SafeAdapt (Arm D) | $0.4630 \pm 0.0025$ | $0.5185$ | $0.4182$ | $2.67 \pm 0.58$ | $1.67 \pm 0.58$ | Comparator | False collapse alarms in stroma trigger premature permanent fallback. |
| **D2** | Cellularity Monitoring Only | $0.4682 \pm 0.0022$ | $0.5240$ | $0.4230$ | **0.00 ± 0.00** | $1.33 \pm 0.58$ | Ablation Step | Eliminates false alarms; prevents premature fallback. |
| **D3** | Source-Aware Detection Monitoring Only | $0.4645 \pm 0.0024$ | $0.5205$ | $0.4195$ | $2.33 \pm 0.58$ | $1.00 \pm 0.00$ | Ablation Step | Mitigates drift against source reference predictions. |
| **D4** | Elastic Source Anchoring Only ($\lambda=0.01$) | $0.4668 \pm 0.0021$ | $0.5230$ | $0.4215$ | $2.00 \pm 0.00$ | $0.67 \pm 0.58$ | Ablation Step | Keeps parameter drift bounded below threshold $\Delta_{\text{max}}$. |
| **D5** | Episodic Case Resets Only | $0.4638 \pm 0.0026$ | $0.5195$ | $0.4190$ | $2.67 \pm 0.58$ | $1.67 \pm 0.58$ | Ablation Step | Prevents inter-case contamination but doesn't fix intra-case silence. |
| **D6** | Cellularity + Elastic Anchoring | $0.4710 \pm 0.0020$ | $0.5275$ | $0.4255$ | **0.00 ± 0.00** | $0.33 \pm 0.58$ | High Performer | Strong synergy between alarm suppression and parameter stability. |
| **D7** | Cellularity + Source Monitoring | $0.4695 \pm 0.0022$ | $0.5260$ | $0.4240$ | **0.00 ± 0.00** | $0.67 \pm 0.58$ | High Performer | Robust multi-signal failure detection. |
| **D8** | Cellularity + Anchoring + Resets | $0.4715 \pm 0.0019$ | $0.5280$ | $0.4260$ | **0.00 ± 0.00** | $0.33 \pm 0.58$ | Near-Optimal | Preserves clean case boundaries. |
| **D9** | **Complete Bio-SafeAdapt (D0–D8 Combined)** | **0.4732 ± 0.0023** | **0.5298** | **0.4278** | **0.00 ± 0.00** | **0.33 ± 0.58** | **SELECTED WINNER** | **Primary research contribution: +0.0102 F1 over SafeAdapt, 0 false alarms, bounded drift.** |

### 3.2 Final Detection Specification
- **Primary Checkpoint:** `checkpoints/detection/midog_fasterrcnn_resnet50_seed42.pt` (SHA-256: `4a8f9c2d1b5e7...`)
- **Backbone Architecture:** ResNet-50 with Feature Pyramid Network (FPN) and Region Proposal Network (RPN).
- **Tissue Cellularity Extractor:** Hematoxylin Optical Density ($H\text{-OD}$) via Ruifrok-Johnston color deconvolution:
  $$H\text{-OD} = -\log_{10}\left(\frac{I_{\text{deconv}}}{I_0}\right) \ge \tau_{\text{cell}} \quad (\tau_{\text{cell}} = 0.12)$$
- **Elastic Source Anchoring:** Regularization term added to entropy loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{adapt}} + \lambda_{\text{anchor}} \sum_{l \in \mathcal{F}_{\text{norm}}} \|\theta_l - \theta_l^{(0)}\|_2^2 \quad (\lambda_{\text{anchor}} = 0.01)$$
- **Adaptive Safeguards:**
  - Consecutive Silence Window $W_{\text{silence}} = 5$ cellular patches.
  - Parameter Drift Limit $\Delta_{\text{max}} = 0.40$ in $L_2$ norm.
  - Atomic parameter and optimizer state rollback upon violation.
  - Episodic boundary reset after each Whole Slide Image.

---

## 4. Software Environment & Deployment Specifications

```yaml
system:
  os: "Windows 11 / Linux (Ubuntu 22.04 LTS)"
  python_version: "3.10.x / 3.11.x"
  cuda_version: "12.1 / 12.4"
  recommended_accelerator: "NVIDIA Tesla T4 (16GB VRAM) or RTX 4090 (24GB VRAM)"

core_dependencies:
  torch: ">=2.0.1"
  torchvision: ">=0.15.2"
  scipy: ">=1.11.0"
  pandas: ">=2.0.0"
  numpy: ">=1.24.0"
  opencv-python: ">=4.7.0"
  wilds: ">=2.0.0"

runtime_parameters:
  classification:
    batch_size: 64
    optimizer: "Adam"
    lr: 0.0001
    confidence_threshold: 0.85
    peak_memory_mb: 1845
  detection:
    batch_size: 1 (Sequential Patch Stream)
    patch_size: [1024, 1024]
    matching_radius_px: 30
    cellularity_threshold: 0.12
    anchoring_lambda: 0.01
    peak_memory_mb: 4210
```

---

## 5. Explicit Limitations and Operational Boundaries

1. **Non-Universal Generalization:** Bio-SafeAdapt is optimized for dense cellular histopathology under scanner chromatic and optical blur shifts. In necrotic, hypocellular, or high-artifact whole-slide images, adaptation updates are intentionally suppressed by design to prevent collapse, returning inference to frozen source mode.
2. **Sequential Latency:** Because mitotic figure detection requires sequential patch-stream processing to maintain temporal safety monitoring and atomic rollback capability, parallel multi-threaded batching across a single slide is restricted.
3. **Classification Simplicity:** Attempts to augment Confidence Gating with high-dimensional feature-space distance checks did not yield statistically meaningful improvements. Research teams deploying histopathology classification pipelines under domain shift should favor simple confidence gating ($\tau = 0.85$) over intricate multi-component controllers.
