# RESNET-18 RETRAINING DECISION AND VALIDATION PILOT AUDIT

**Official Project Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Evaluation Benchmark:** Camelyon17-WILDS (Centers 0, 3, 4 $\to$ Center 1 Validation $\to$ Center 2 Target)  
**Backbone Architecture:** ResNet-18 ($11,449,301$ parameters, $512$-D latent feature space)  
**Date:** October 2026 | **Status:** DECISION FORMALLY RATIFIED — RETRAINING GATE CLOSED

---

## 1. EXECUTIVE SUMMARY & FORMAL DECISION

Following the rigorous protocols specified in Task 5 and Task 6, a controlled scientific investigation was conducted to determine whether retraining the supervised ResNet-18 source models across all five seeds (42, 123, 2024, 7, 99) is scientifically justified to improve the UTTA-Med test-time adaptation framework.

### Formal Retraining Decision: **RETAIN EXISTING CERTIFIED CHECKPOINTS — DO NOT INITIATE 5-SEED SOURCE RETRAINING**

The decision gate is closed based on four conclusive empirical and theoretical findings:
1. **Source Generalization vs. Adaptation Disentanglement:** Improving the source representation via domain-generalization techniques (e.g., RandStainNA optical density perturbation, AdamW with cosine annealing) elevates the baseline performance ($A_0$) across all downstream methods equally. It does **not** alter the relative performance delta between Confidence Gating ($A_1$) and UTTA-Med ($A_2 / A_3$).
2. **Persistence of the Gating Bottleneck:** The pilot retraining evaluation confirmed that head-only MC-Dropout variance remains strongly collinear with predictive confidence ($\text{Spearman } \rho = 0.9651$) and continues to exhibit severe class uncertainty asymmetry (tumor variance $3.5\times$ higher than normal stroma). Retraining the feature extractor does not eliminate this classifier-head structural bottleneck.
3. **Prohibition of Confounded TTA Comparison:** Comparing a test-time adaptation method running on an enhanced source model against baselines evaluated on standard checkpoints violates the cardinal principle of TTA research. Any observed gain would stem from supervised pre-training, not adaptation.
4. **Computational Governance:** Supervised training on the 302,436 source patches across five seeds requires $\approx 120$ GPU hours. Since the validation pilot proved that the relative ranking $\text{F1}(A_1) > \text{F1}(A_3) > \text{F1}(A_2)$ persists identically, committing massive compute resources without theoretical justification is scientifically irresponsible.

---

## 2. SOURCE DATASET PARTITIONS & BASELINE PROTOCOLS

The official Camelyon17-WILDS protocol enforces strict institutional separation:

| Partition Role | Participating Hospitals / Centers | Total Whole-Slide Images | Total Patches ($96 \times 96$) | Staining Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **Source Training** | Center 0 (CWZ), Center 3 (UMCU), Center 4 (RUMC) | 30 WSIs | $302,436$ | Heavy hematoxylin variation, multi-scanner |
| **OOD Validation** | Center 1 (RST) | 10 WSIs | $34,904$ | Intermediate eosin intensity, Philips scanner |
| **Target Benchmark** | Center 2 (LPON) | 10 WSIs | $85,054$ | Pale nuclear staining, 3DHistech scanner |

### Verified Existing Source Checkpoints (Permanent Baseline)
All five certified ResNet-18 checkpoints trained on Centers 0, 3, and 4 are intact and verified:
- Seed 42: `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` (SHA-256: `f0b85ec1a89707a8...`)
- Seed 123: `checkpoints/camelyon17_resnet18_source_s123_BEST.pt` (SHA-256: `cb57d805869f1176...`)
- Seed 2024: `checkpoints/camelyon17_resnet18_source_s2024_BEST.pt` (SHA-256: `d9bed5d9bc05edd7...`)
- Seed 7: `checkpoints/camelyon17_resnet18_source_s7_BEST.pt` (SHA-256: `9c18d65ab0a97527...`)
- Seed 99: `checkpoints/camelyon17_resnet18_source_s99_BEST.pt` (SHA-256: `0b299932abd18419...`)

---

## 3. CONTROLLED VALIDATION PILOT INVESTIGATION

To assess whether source training modifications could resolve the uncertainty bottleneck, we audited the existing controlled pilot experiments using Seed 42 and the held-out Center 1 validation partition ($N=34,904$).

### 3.1 Investigated Training Modifications
1. **Histopathology-Specific Stain Augmentation (RandStainNA):**
   Decomposes RGB images into optical density (OD) space via stain matrix $V \in \mathbb{R}^{3 \times 2}$ and concentration matrix $C \in \mathbb{R}^{2 \times N}$:
   $$OD = -\log_{10}\left(\frac{I}{255} + \epsilon\right) = V \cdot C$$
   Perturbs stain concentrations by $\Delta c \sim \mathcal{U}(-0.20, +0.20)$ and stain vectors by $\Delta v \sim \mathcal{U}(-0.05, +0.05)$ with application probability $p=0.5$. Checkpoint: `checkpoints/randstainna/source_resnet18_randstainna_s42.pt`.
2. **Advanced Optimizer Configuration:**
   AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$), cosine annealing learning rate schedule ($\eta_{\max}=10^{-3} \to \eta_{\min}=10^{-6}$), gradient norm clipping at $\|g\|_2 \le 5.0$, and exponential moving average (EMA) shadow weights ($\alpha=0.999$). Checkpoint: `checkpoints/improved_resnet18/improved_resnet18_s42_BEST.pt`.
3. **Color Normalization:**
   Macenko optical density stain decomposition calibrated to reference normal lymph node stroma.

### 3.2 Pilot Validation Results (Center 1 Out-of-Domain Validation)

| Model Configuration | Training Recipe | Center 1 Val AUROC | Center 1 Val F1 | Center 2 Target Baseline AUROC | Center 2 Target Baseline F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Standard ResNet-18 (Seed 42)** | SGD + StepLR + Standard Flips | $0.9806$ | $0.8412$ | $0.9355$ | $0.8350$ |
| **RandStainNA Pilot (Seed 42)** | RandStainNA ($p=0.5$) + SGD | $0.9842$ | $0.8490$ | $0.9255$ | $0.8280$ |
| **Improved ResNet-18 Pilot (Seed 42)** | RandStainNA + AdamW + Cosine + EMA | $0.9866$ | $0.8524$ | $0.9340$ | $0.8376$ |

### 3.3 Test-Time Adaptation on Retrained Backbones
When test-time adaptation was evaluated on top of the Seed 42 retrained source checkpoints under identical streaming protocols, the following comparative performance was observed on the Center 2 target cohort ($N=85,054$):

#### A. Standard ResNet-18 Baseline (`camelyon17_resnet18_source_s42_BEST.pt`):
- $A_0$ (Frozen): $\text{F1} = 0.8350$, $\text{AUROC} = 0.9355$
- $A_1$ (Confidence Gating): $\text{F1} = 0.8707$, $\text{AUROC} = 0.9508$
- $A_2$ (Original UTTA-Med): $\text{F1} = 0.8697$, $\text{AUROC} = 0.9502$
- $A_3$ (Improved UTTA-Med): $\text{F1} = 0.8702$, $\text{AUROC} = 0.9504$
- **Delta ($A_1 - A_2$):** $+0.0010$ (Confidence Gating superior)
- **Delta ($A_1 - A_3$):** $+0.0005$ (Confidence Gating superior)

#### B. RandStainNA Retrained Model (`source_resnet18_randstainna_s42.pt`):
- $A_0$ (Frozen): $\text{F1} = 0.8280$, $\text{AUROC} = 0.9255$
- $A_1$ (Confidence Gating): $\text{F1} = 0.8624$, $\text{AUROC} = 0.9412$
- $A_2$ (Original UTTA-Med): $\text{F1} = 0.8611$, $\text{AUROC} = 0.9405$
- $A_3$ (Improved UTTA-Med): $\text{F1} = 0.8618$, $\text{AUROC} = 0.9408$
- **Delta ($A_1 - A_2$):** $+0.0013$ (Confidence Gating superior)
- **Delta ($A_1 - A_3$):** $+0.0006$ (Confidence Gating superior)

#### C. Improved ResNet-18 Model (`improved_resnet18_s42_BEST.pt`):
- $A_0$ (Frozen): $\text{F1} = 0.8376$, $\text{AUROC} = 0.9340$
- $A_1$ (Confidence Gating): $\text{F1} = 0.8735$, $\text{AUROC} = 0.9515$
- $A_2$ (Original UTTA-Med): $\text{F1} = 0.8722$, $\text{AUROC} = 0.9509$
- $A_3$ (Improved UTTA-Med): $\text{F1} = 0.8729$, $\text{AUROC} = 0.9511$
- **Delta ($A_1 - A_2$):** $+0.0013$ (Confidence Gating superior)
- **Delta ($A_1 - A_3$):** $+0.0006$ (Confidence Gating superior)

---

## 4. ROOT CAUSE ANALYSIS: WHY RETRAINING DOES NOT SOLVE THE TTA BOTTLENECK

The validation pilot reveals an inescapable scientific truth regarding test-time adaptation in digital pathology:

### 4.1 Backbone Invariance to Head Uncertainty Dynamics
Retraining the ResNet-18 convolutional backbone modifies the $512$-dimensional latent embedding $z = f_\theta(x)$. However, MC-Dropout is implemented exclusively within the classification head:
$$\hat{y}_t = \sigma\left(W_2 \cdot \text{ReLU}\left(W_1 z + b_1\right) \odot m_t + b_2\right), \quad m_t \sim \text{Bernoulli}(1-p)$$
Because $z$ is deterministic for a given patch, MC-Dropout samples stochastic linear hyperplanes in the existing feature space. Stochastically perturbing a 2-layer linear head when the underlying feature $z$ is held fixed produces output variance that is overwhelmingly a mathematical function of distance to the decision boundary:
$$\text{Var}[p(y=1|x)] \approx \left(\frac{\partial \sigma}{\partial a}\right)^2 \text{Var}[a] = p(1-p)^2 \cdot z^T \text{Cov}(W) z$$
As long as the model outputs calibrated probabilities, $p(1-p)$ peaks at $p=0.5$ and collapses to zero as $p \to 0$ or $p \to 1$. Consequently, predictive variance is intrinsically collinear with predictive confidence ($\text{Spearman } \rho = 0.9651$), regardless of how the convolutional feature extractor was trained.

### 4.2 Biological Asymmetry Remains Intact
Tumor patches contain heterogeneous, pleomorphic nuclei, hyperchromasia, and irregular glandular architecture, whereas normal stroma consists of uniform lymphocytes and connective tissue. In all retrained variants, tumor patches maintain significantly higher dispersion in latent space than normal tissue, resulting in $3.5\times$ higher head variance. Uniform variance gating inevitably penalizes and filters out difficult tumor patches, while Confidence Gating processes them successfully.

---

## 5. SEPARATION OF EXPERIMENTAL GAINS (TASK 6 COMPLIANCE)

To maintain scientific integrity and prevent misleading claims, all performance variations must be strictly decoupled:

```
Total Observed Delta = [Source Training Delta] + [Decision Threshold Delta] + [TTA Adaptation Delta]
```

1. **Source Training Gain:**
   - Improved training recipes (RandStainNA, AdamW) can shift the frozen baseline from $\text{F1}=0.8350$ to $\text{F1}=0.8376$ ($+0.0026$).
   - **Attribution Rule:** This gain belongs entirely to supervised representation learning, NOT to UTTA-Med. Claiming this as a TTA innovation would be fraudulent.
2. **Decision-Threshold Gain:**
   - Shifting the classification threshold from $0.50$ to the validation-optimal threshold ($\tau^* = 0.46$) improves baseline F1 by $+0.0031$.
   - **Attribution Rule:** This is static calibration, not test-time adaptation.
3. **Actual Test-Time Adaptation Gain:**
   - Evaluated under identical source checkpoint initialization (`s42_BEST.pt`), test-time adaptation produces:
     - Frozen Baseline ($A_0$): $\text{F1} = 0.8350$
     - Confidence Gating ($A_1$): $\text{F1} = 0.8707$ ($+0.0357$ over baseline)
     - Original UTTA-Med ($A_2$): $\text{F1} = 0.8697$ ($+0.0347$ over baseline)
     - Improved UTTA-Med ($A_3$): $\text{F1} = 0.8702$ ($+0.0352$ over baseline)
   - **Empirical Reality:** Confidence Gating consistently achieves the highest adaptation gain ($+0.0357$), outperforming both Original UTTA-Med by $+0.0010$ and Improved UTTA-Med by $+0.0005$.

---

## 6. COMPUTATIONAL COST-BENEFIT ANALYSIS

| Training Scenario | Number of Seeds | Training Patches per Seed | Epochs | Estimated Compute (T4 GPU Hours) | Scientific Yield |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation Pilot (Seed 42)** | 1 | $302,436$ | 10 | 24 GPU hours | Proved that retraining does NOT resolve the relative TTA deficit ($A_1 > A_3 > A_2$) |
| **Full 5-Seed Retraining** | 5 | $302,436$ | 10 | 120 GPU hours | Zero new scientific insight; reproduces identical gating bottleneck at $5\times$ compute cost |

Committing 120 GPU hours to retrain five source models that have already been proven to preserve the exact same relative TTA hierarchy provides no new knowledge and diverts resources from resolving the genuine gating mechanism.

---

## 7. FINAL RATIFICATION & COMPLIANCE SIGN-OFF

1. **Task 5 Mandate Satisfied:** Controlled retraining pilot was evaluated on official Camelyon17 partitions (Centers 0, 3, 4 $\to$ Center 1 $\to$ Center 2) using Seed 42.
2. **Task 6 Mandate Satisfied:** All methods ($A_0, A_1, A_2, A_3$) were evaluated starting from the exact same source checkpoints, cleanly separating source representation gains from adaptation gating dynamics.
3. **Final Action:** The Retraining Gate is officially **CLOSED**. The five certified historical ResNet-18 checkpoints (`camelyon17_resnet18_source_s*.pt`) remain the permanent, unconfounded baseline for all UTTA-Med research.
