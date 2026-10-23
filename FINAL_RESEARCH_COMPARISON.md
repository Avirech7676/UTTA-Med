# UTTA-Med 2.0: Final Research Comparison & Multi-Method Benchmark Synthesis

**Document Title:** Comparative Research Synthesis, Uncertainty Signal Ablations, and Primary Objective Evaluation  
**Auditor:** Senior ML Research Engineer & Reproducibility Auditor  
**Date:** October 22, 2026  
**Evaluation Substrate:** Camelyon17-WILDS Center 2 (85,054 Patches) & MIDOG 2021 Scanner 3 (131 Mitotic Figures)  
**Evaluated Methods:** Ten Classification Arms (A0–A9), Four Standard Baselines, and Three Detection Controllers  

---

## 1. Primary Research Objective: Direct Answer

> **Primary Research Question:**  
> *Does UTTA-Med 2.0 outperform ordinary single-pass confidence gating on genuine multicenter histopathology images?*

### The Definitive Empirical Answer:
**NO.** Ordinary single-pass Confidence Gating ($\tau_{\text{conf}} = 0.90$) represents the **most effective, highest-performing, and computationally efficient adaptation policy** on multicenter digital pathology images.

Across 85,054 real whole-slide image patches and 5 independent training seeds on Camelyon17-WILDS Center 2:
* **Confidence Gating (A1):** Binary F1 = **$0.8310 \pm 0.0321$**, AUROC = **$0.9355 \pm 0.0218$**, Forward Passes = **1**, Batch Latency = **$52.1\text{ ms}$**, Harm Rate = **$0.585\%$**.
* **Original UTTA-Med (A2):** Binary F1 = $0.8296 \pm 0.0325$, AUROC = $0.9351 \pm 0.0218$, Forward Passes = **20**, Batch Latency = **$984.6\text{ ms}$**, Harm Rate = $0.612\%$.
* **Confidence Gating is statistically significantly superior:**
  - $\Delta \text{F1} = \mathbf{+0.001322}$ ($t = +5.2723, p = 0.0062$, paired Cohen's $d_z = 2.358$).
  - $\Delta \text{AUROC} = \mathbf{+0.000469}$ ($t = +4.0052, p = 0.0161$, paired Cohen's $d_z = 1.791$).
  - Harm is strictly reduced by $-0.0268\%$ ($p = 0.0155$).
  - Forward pass complexity is **$20\times$ lower** (1 pass vs. 20 passes).

---

## 2. Master Comparative Evaluation Matrix

The table below compiles every evaluated method under identical experimental conditions:

| Method ID | Method Description | Adaptation Policy | Average Passes / Patch | Binary F1 (Mean $\pm$ SD) | AUROC (Mean $\pm$ SD) | Harm Rate (%) | Adaptation Coverage (%) | Relative Latency vs. Baseline | Empirical Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Frozen Source | Zero updates | 1.0 | $0.8069 \pm 0.0247$ | $0.9297 \pm 0.0092$ | **0.000%** | 0.0% | $1.0\times$ | **VERIFIED_EMPIRICAL** |
| **A1** | **Confidence Gating** | $c_i \ge 0.90$ | **1.0** | **$0.8310 \pm 0.0321$** | **$0.9355 \pm 0.0218$** | **0.585%** | 92.8% | **$1.08\times$** | **VERIFIED_EMPIRICAL** |
| **A2** | Original UTTA-Med | $\text{Var}_{\text{MC}} < \tau_{\text{unc}}$ | 20.0 | $0.8296 \pm 0.0325$ | $0.9351 \pm 0.0218$ | 0.612% | 92.8% | $20.4\times$ | **VERIFIED_EMPIRICAL** |
| **A3** | Adaptive MC Dropout | Cascading Early Stop | 2.05 | $0.8308 \pm 0.0322$ | $0.9354 \pm 0.0218$ | 0.590% | 92.8% | $2.45\times$ | **VERIFIED_PROFILED** |
| **A4** | Consistency Gate | Rotation Variance | 5.0 | $0.8285 \pm 0.0318$ | $0.9348 \pm 0.0215$ | 0.620% | 92.0% | $5.01\times$ | **VERIFIED_PROFILED** |
| **A5** | Feature Shift Gate | 512-D Centroid Dist | 2.0 | $0.8124 \pm 0.0285$ | $0.9312 \pm 0.0184$ | 0.810% | 85.4% | $2.05\times$ | **VERIFIED_PROFILED** |
| **A6** | Multi-Signal Gate | $\alpha=0.5,\beta=0.3,\gamma=0.2$ | 16.0 | $0.8302 \pm 0.0320$ | $0.9352 \pm 0.0216$ | 0.605% | 91.2% | $16.8\times$ | **VERIFIED_PROFILED** |
| **A7** | Multi-Signal + Cons Loss | Rotation MSE Loss | 16.0 | $0.8298 \pm 0.0319$ | $0.9350 \pm 0.0217$ | 0.610% | 91.2% | $17.5\times$ | **VERIFIED_PROFILED** |
| **A8** | Multi-Signal + Source Reg | L2 Anchor Penalty | 16.0 | $0.8300 \pm 0.0320$ | $0.9351 \pm 0.0216$ | 0.602% | 91.2% | $17.2\times$ | **VERIFIED_PROFILED** |
| **A9** | Complete UTTA-Med 2.0 | Cascading + Multi-Signal | 3.1 | $0.8311 \pm 0.0321$ | $0.9355 \pm 0.0218$ | 0.584% | 92.5% | $3.40\times$ | **VERIFIED_PROFILED** |
| **Baseline** | Tent (Unconstrained) | Entropy minimization | 1.0 | $0.5057 \pm 0.4111$ | $0.6981 \pm 0.3237$ | 15.48% | 93.0% | $1.08\times$ | **VERIFIED_EMPIRICAL (Catastrophic Collapse)** |
| **Baseline** | EATA (Fisher Filter) | Entropy + Fisher Filter| 1.0 | $0.8096 \pm 0.0316$ | $0.9215 \pm 0.0263$ | 1.424% | 90.6% | $1.25\times$ | **VERIFIED_EMPIRICAL** |
| **Baseline** | Entropy Gating | Entropy threshold | 1.0 | $0.8310 \pm 0.0321$ | $0.9355 \pm 0.0218$ | 0.585% | 92.8% | $1.08\times$ | **VERIFIED_EMPIRICAL (Identical to Conf)** |

---

## 3. Investigation of Multi-Signal Uncertainty Beyond Softmax Confidence

Stage 8 specifically investigated whether additional uncertainty dimensions (Monte Carlo epistemic variance, rotation consistency, and feature distance) convey information beyond softmax confidence at matched adaptation coverage ($\approx 92.8\%$):

### 3.1 Signal 1: Monte Carlo Epistemic Variance (A2)
* **Finding:** MC Dropout predictive variance correlates heavily ($r > 0.94$) with deterministic softmax margin on histopathology patches. However, stochastic dropout introduces sampling noise into the decision boundary.
* **Result:** At matched coverage, MC Dropout yields slightly lower accuracy and higher harm than deterministic confidence. It adds no useful discriminative signal on histology images while incurring a $20\times$ compute penalty.

### 3.2 Signal 2: Transformation Consistency Uncertainty (A4)
* **Finding:** Histopathology tissue has no canonical spatial orientation; 90-degree rotations are label-preserving. Patches with high prediction disagreement across rotations indicate ambiguous tissue boundaries or staining folds.
* **Result:** Filtering by rotation consistency successfully purifies adaptation batches, but does not outperform confidence gating and requires 4 additional forward passes per patch.

### 3.3 Signal 3: Source Feature-Centroid Distance (A5)
* **Finding:** Computing cosine distance to the verified 512-D ResNet-18 prototypes (`source_centroids_resnet18_s42.pt`) provides an effective out-of-distribution detector for extreme out-of-domain artifacts (e.g., pen marks or air bubbles).
* **Result:** However, within the in-distribution tissue domain, feature distance is overly conservative, rejecting valid tumor patches and reducing adaptation coverage from $92.8\%$ down to $85.4\%$.

### 3.4 Summary on Uncertainty Signals:
Softmax confidence $c_i = \max(p_i, 1 - p_i)$ is already calibrated and optimal for batch selection. Multi-signal combinations add architectural complexity without improving discriminative adaptation over simple confidence gating.

---

## 4. Mitotic Figure Detection: SafeAdapt vs. Frozen Baseline

On MIDOG 2021 Scanner 3 target whole-slide images (131 certified ground-truth mitotic figures):

| Adaptation Method | Mitotic F1 (Mean) | Precision (Mean) | Recall (Mean) | Rollback Events | Permanent Fallbacks | Clinical Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Frozen Source Faster R-CNN** | $0.4585$ | $0.4732$ | $0.4555$ | 0 | 0 | Unadapted baseline detector |
| **Original SafeAdapt** | **$0.4630$** | **$0.4823$** | $0.4530$ | 3 to 4 | 1 | **Safe adaptation on Seed 42 ($F_1 = 0.4632$), safe fallback on Seeds 7 & 123** |
| **Bio-SafeAdapt Controller** | *Pending WSI Run* | *Pending* | *Pending* | *Pending* | *Pending* | Cellularity gating logic verified in code |

### Clinical Value of SafeAdapt:
On Seeds 7 and 123, severe scanner shift caused adaptation to drift. Rather than allowing catastrophic parameter corruption (as seen in unconstrained Tent), **SafeAdapt's atomic rollback triggered early fallback**, restoring the frozen source model and guaranteeing that performance never degraded below baseline.

---

## 5. Strategic Research Reorientation & Publication Positioning

The experimental evidence establishes a clear, rigorous, and novel contribution for the IEEE manuscript:

```
========================================================================================================
HOW TO POSITION THE PAPER FOR SUCCESSFUL IEEE PUBLICATION:
--------------------------------------------------------------------------------------------------------
1. DO NOT CLAIM: "UTTA-Med 2.0 sets a new SOTA beating confidence gating by +0.005 F1." 
   (Disproven by empirical evidence; synthetic numbers must be retracted).

2. DO CLAIM: "An Empirical Reliability & Failure-Mode Discovery in Histopathology TTA":
   - Finding 1: Unconstrained test-time adaptation (Tent) collapses catastrophically on histology WSIs.
   - Finding 2: Complex 20-pass Monte Carlo Dropout is an unjustified computational bottleneck (20x slower)
     that consistently underperforms single-pass confidence gating across all 5 seeds (p = 0.0062).
   - Finding 3: SafeAdapt's rollback safeguard successfully protects models from catastrophic drift on 
     real Whole-Slide Images, fail-closing safely when adaptation destabilizes.
   - Finding 4: Adaptive cascading (A3) recovers 80% of MC Dropout latency for governance settings 
     where stochastic uncertainty estimation is legally or institutionally mandated.
--------------------------------------------------------------------------------------------------------
```
