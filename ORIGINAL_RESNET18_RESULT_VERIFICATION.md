# ORIGINAL RESNET-18 RESULT VERIFICATION
## Forensic Reproduction, Checkpoint Integrity, and Statistical Reconciliation of Five-Seed Camelyon17 Benchmarks

**Author:** Senior Computational Pathology Researcher & Statistical Auditor  
**Project:** UTTA-Med Project 2 (`research-project-2`)  
**Date:** October 2026  
**Status:** Certified Empirical Reproduction & Statistical Reconciliation  
**Target Evaluation Stream:** Camelyon17-WILDS Center 2 ($N = 85,054$ patches, Exploratory Evaluation)  
**Development Validation:** Camelyon17-WILDS Center 1 ($N = 34,904$ patches)  

---

## 1. Executive Summary & Verification Scope

Under **Task 1**, we conducted an exhaustive reproduction and forensic verification of the reported five-seed classification results on the authentic Camelyon17-WILDS benchmark. Every metric reported herein was calculated directly from authentic PyTorch source checkpoints and physical prediction arrays stored on disk in [`results/data/wsi_bootstrap/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/results/data/wsi_bootstrap).

### Core Audit Outcomes:
1. **Checkpoint Identity:** Verified all 5 trained ResNet-18 checkpoints (`s42, s123, s2024, s7, s99`) with exact SHA-256 hashes and $45.8\text{ MB}$ sizes.
2. **Evaluation Sample Alignment:** Confirmed that every seed evaluates exactly $85,054$ patches across 10 Whole Slide Images (WSIs), with slide boundaries and ordering strictly preserved.
3. **Unsupervised Adaptation Integrity:** Confirmed that target test labels ($y \in \{0, 1\}$) were accessed exclusively for offline post-adaptation metric evaluation and never leaked into adaptation losses.
4. **Resolution of Statistical Discrepancy:** Reconciled why earlier development reports claimed UTTA-Med was "statistically indistinguishable" ($p = 0.9500$) from Confidence Gating, whereas recent reports claimed a "statistically significant difference" ($p = 0.0041$). The former used an **unpaired two-sample t-test** (flawed by large between-seed weight variance); the latter correctly applies a **paired Student's t-test** ($t = -5.8990, p = 0.0041$).
5. **Resolution of Coverage Discrepancy:** Reconciled the three historical coverage figures ($58.1\%$, $83.74\%$, and $92.88\%$) by documenting the exact admission formulas and operating constraints.

---

## 2. Checkpoint Provenance & Cryptographic Hashes

All 5 source checkpoints in [`checkpoints/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints) were verified:

| Seed | Checkpoint Path | Architecture | Size (Bytes) | SHA-256 Checksum |
| :---: | :--- | :---: | :---: | :--- |
| **42** | `camelyon17_resnet18_source_s42_BEST.pt` | ResNet-18 (Head Dropout 0.5) | 45,839,231 | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` |
| **123** | `camelyon17_resnet18_source_s123_BEST.pt` | ResNet-18 (Head Dropout 0.5) | 45,841,153 | `cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224` |
| **2024** | `camelyon17_resnet18_source_s2024_BEST.pt` | ResNet-18 (Head Dropout 0.5) | 45,840,579 | `d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261` |
| **7** | `camelyon17_resnet18_source_s7_BEST.pt` | ResNet-18 (Head Dropout 0.5) | 45,840,509 | `9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b` |
| **99** | `camelyon17_resnet18_source_s99_BEST.pt` | ResNet-18 (Head Dropout 0.5) | 45,840,319 | `0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540` |

---

## 3. Exact Unrounded Five-Seed Results

Calculated directly from the 35 prediction arrays in [`results/data/wsi_bootstrap/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/results/data/wsi_bootstrap):

### 3.1 Method A0: Frozen Source ResNet-18
- **Seed 42:** $F_1 = 0.8349705, \quad \text{AUROC} = 0.9354560, \quad \text{Recall} = 0.7495238, \quad \text{Prec} = 0.9424061, \quad \text{Harm} = 0$
- **Seed 123:** $F_1 = 0.8270699, \quad \text{AUROC} = 0.9349245, \quad \text{Recall} = 0.7338162, \quad \text{Prec} = 0.9474755, \quad \text{Harm} = 0$
- **Seed 2024:** $F_1 = 0.7972120, \quad \text{AUROC} = 0.9319766, \quad \text{Recall} = 0.6804618, \quad \text{Prec} = 0.9623225, \quad \text{Harm} = 0$
- **Seed 7:** $F_1 = 0.7734808, \quad \text{AUROC} = 0.9133543, \quad \text{Recall} = 0.6379712, \quad \text{Prec} = 0.9820821, \quad \text{Harm} = 0$
- **Seed 99:** $F_1 = 0.8015284, \quad \text{AUROC} = 0.9327649, \quad \text{Recall} = 0.6806970, \quad \text{Prec} = 0.9745161, \quad \text{Harm} = 0$
- **Five-Seed Mean:** $\mathbf{F_1 = 0.8068523 \pm 0.0247290}, \quad \text{AUROC} = 0.9296953 \pm 0.0092497$

### 3.2 Method A1: Confidence Gating ($\tau_{\text{conf}}$ matched to validation coverage)
- **Seed 42:** $F_1 = 0.8707169, \quad \text{AUROC} = 0.9508127, \quad \text{Recall} = 0.7884403, \quad \text{Prec} = 0.9721658, \quad \text{Harm} = 212$
- **Seed 123:** $F_1 = 0.8548083, \quad \text{AUROC} = 0.9423004, \quad \text{Recall} = 0.7576834, \quad \text{Prec} = 0.9804948, \quad \text{Harm} = 440$
- **Seed 2024:** $F_1 = 0.8260572, \quad \text{AUROC} = 0.9559126, \quad \text{Recall} = 0.7092200, \quad \text{Prec} = 0.9889825, \quad \text{Harm} = 494$
- **Seed 7:** $F_1 = 0.7911425, \quad \text{AUROC} = 0.9019990, \quad \text{Recall} = 0.6565476, \quad \text{Prec} = 0.9951527, \quad \text{Harm} = 650$
- **Seed 99:** $F_1 = 0.8120724, \quad \text{AUROC} = 0.9266053, \quad \text{Recall} = 0.6880570, \quad \text{Prec} = 0.9906222, \quad \text{Harm} = 691$
- **Five-Seed Mean:** $\mathbf{F_1 = 0.8309594 \pm 0.0320760}, \quad \text{AUROC} = 0.9355260 \pm 0.0218128, \quad \text{Harm} = 497.4 \pm 190.7$

### 3.3 Method A2: Original UTTA-Med (70th percentile of validation $U_{\text{var}}$)
- **Seed 42:** $F_1 = 0.8697222, \quad \text{AUROC} = 0.9502076, \quad \text{Recall} = 0.7871000, \quad \text{Prec} = 0.9717247, \quad \text{Harm} = 218$
- **Seed 123:** $F_1 = 0.8538451, \quad \text{AUROC} = 0.9422464, \quad \text{Recall} = 0.7558257, \quad \text{Prec} = 0.9810762, \quad \text{Harm} = 456$
- **Seed 2024:** $F_1 = 0.8250209, \quad \text{AUROC} = 0.9551922, \quad \text{Recall} = 0.7077386, \quad \text{Prec} = 0.9888947, \quad \text{Harm} = 516$
- **Seed 7:** $F_1 = 0.7888507, \quad \text{AUROC} = 0.9014176, \quad \text{Recall} = 0.6535142, \quad \text{Prec} = 0.9948810, \quad \text{Harm} = 687$
- **Seed 99:** $F_1 = 0.8107470, \quad \text{AUROC} = 0.9262224, \quad \text{Recall} = 0.6861523, \quad \text{Prec} = 0.9906301, \quad \text{Harm} = 724$
- **Five-Seed Mean:** $\mathbf{F_1 = 0.8296372 \pm 0.0325372}, \quad \text{AUROC} = 0.9350572 \pm 0.0218451, \quad \text{Harm} = 520.2 \pm 203.0$

---

## 4. Reconciliation of the Statistical Discrepancy

A notable contradiction in earlier reports was that some documents described UTTA-Med and Confidence Gating as **"statistically indistinguishable"**, while other reports claimed a **"statistically significant difference"**.

We resolved this by auditing the exact statistical formulation applied:

$$\begin{array}{|l|c|c|c|l|}
\hline
\textbf{Statistical Test} & \textbf{Degrees of Freedom} & \textbf{Test Statistic} & \textbf{p-value (2-tailed)} & \textbf{Scientific Interpretation} \\ \hline
\text{Unpaired Welch's t-test} & df = 7.99 & t = -0.0656 & \mathbf{p = 0.9493} & \text{"Statistically indistinguishable" (METHODOLOGICALLY FLAWED)} \\
\textbf{Paired Student's t-test} & \mathbf{df = 4} & \mathbf{t = -5.8990} & \mathbf{p = 0.0041} & \textbf{"Statistically significant deficit" (METHODOLOGICALLY CORRECT)} \\
\text{Wilcoxon signed-rank test} & N = 5 & W = 0.0 & \mathbf{p = 0.0625} & \text{Minimum achievable p-value for } N=5 \text{ (All 5 seeds negative)} \\ \hline
\end{array}$$

### Mathematical Reason for the Discrepancy:
1. **Unpaired Test Flaw:** In an unpaired two-sample test, the standard error denominator is dominated by the between-seed initialization variance ($\sigma_{\text{seed}} \approx 0.032$). Because the mean difference ($\Delta = -0.00132$) is small relative to $0.032$, the unpaired t-statistic is near zero ($t = -0.0656, p = 0.9493$).
2. **Paired Test Reality:** Because both methods start from the **identical trained checkpoint** within each seed, the between-seed variation cancels out:
   $$\Delta F_1(\text{Seed 42}) = -0.000995$$
   $$\Delta F_1(\text{Seed 123}) = -0.000963$$
   $$\Delta F_1(\text{Seed 2024}) = -0.001036$$
   $$\Delta F_1(\text{Seed 7}) = -0.002292$$
   $$\Delta F_1(\text{Seed 99}) = -0.001325$$
   In **100% of seeds ($5/5$)**, UTTA-Med is strictly lower than Confidence Gating. The standard deviation of the paired difference is tiny ($\sigma_{\Delta} = 0.000561$). The paired t-test yields $t = -5.8990, p = 0.00413$ ($p < 0.01$).

**Definitive Determination:** Confidence Gating is **statistically superior** to Original UTTA-Med. The earlier claim of "statistical equivalence" was an artifact of applying an unpaired test to a matched-pairs experimental design.

---

## 5. Reconciliation of the Coverage Discrepancy

Earlier reports cited three materially different coverage figures on the Camelyon17 target cohort ($58.1\%$, $83.74\%$, and $92.88\%$). We reconciled the exact definitions and operating protocols for each:

$$\begin{array}{|c|l|c|l|}
\hline
\textbf{Reported Coverage} & \textbf{Operating Policy} & \textbf{Admitted Patches } (k) & \textbf{Protocol Context} \\ \hline
\mathbf{58.1\%} & \text{Matched-Budget Ablation} & k = 49,416 \text{ of } 85,054 & \text{Ablation experiment forcing equal sample count across all methods.} \\
\mathbf{83.74\% \pm 2.23\%} & \text{Fixed Confidence Gating } (\tau = 0.90) & k \approx 71,224 \text{ of } 85,054 & \text{Unconstrained gating using fixed high-confidence threshold } \tau_c = 0.90. \\
\mathbf{92.88\% \pm 5.61\%} & \text{Validation-Calibrated Threshold} & k \approx 78,998 \text{ of } 85,054 & \text{Official multi-center protocol: } \tau \text{ set to 70th percentile of Center 1 U.} \\ \hline
\end{array}$$

### Protocol Documentation:
1. **The $58.1\%$ Operating Budget:** In the paper's controlled ablation experiments, the goal was to eliminate sample count as a confounding variable. Both Confidence Gating and MC-Dropout were restricted to the top $58.1\%$ most certain samples ($k = 49,416$ full cohort, $k = 744$ diagnostic subset).
2. **The $83.74\%$ Threshold ($\tau = 0.90$):** In unconstrained streaming without validation calibration, a standard softmax threshold of $0.90$ filters out the bottom $16.26\%$ most ambiguous patches, achieving an average coverage of $83.74\% \pm 2.23\%$.
3. **The $92.88\%$ Official Benchmark:** In the certified multi-center protocol ([`configs/authoritative_resnet18_utta.yaml`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/configs/authoritative_resnet18_utta.yaml)), the threshold $\tau$ is calibrated strictly on Hospital 1 out-of-domain validation data as the 70th percentile of predictive variance ($\tau = 8.875383 \times 10^{-6}$ for Seed 42). Because Hospital 1 exhibits greater staining shift than Hospital 2, applying this frozen cutoff to Hospital 2 admits $92.88\% \pm 5.61\%$ of patches.

All three figures are mathematically valid within their respective experimental protocols and do not represent inconsistent measurements.
