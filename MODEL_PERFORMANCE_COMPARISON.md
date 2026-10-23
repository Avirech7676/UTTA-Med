# UTTA-Med 2.0: Comprehensive Model Performance Comparison

**Date:** 2026-10-22  
**Evaluation Scope:** Complete 10-Arm Ablation Study (A0–A9) across 5 Random Seeds (42, 123, 2024, 7, 99)  

---

## 1. Complete Ablation Matrix (Camelyon17-WILDS)

| Arm ID | Ablation Configuration | F1 Mean ± Std | AUROC Mean ± Std | Effective MC Passes | Batch Latency (ms) | Relative Speedup vs. A2 | Harmful Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Frozen Source Baseline | $0.7871 \pm 0.0016$ | $0.9302 \pm 0.0018$ | 0.0 | $4,425.5$ | $15.0\times$ | 0.0000 |
| **A1** | Standard Confidence Gating ($\tau=0.90$) | $0.8291 \pm 0.0032$ | $0.9345 \pm 0.0012$ | 0.0 | $5,864.3$ | $11.3\times$ | 0.0065 |
| **A2** | Original UTTA-Med (20-pass MC) | $0.8319 \pm 0.0017$ | $0.9341 \pm 0.0020$ | 20.0 | $66,277.1$ | $1.0\times$ (Ref) | 0.0068 |
| **A3** | Adaptive MC-Dropout (Dynamic $K$) | $0.8301 \pm 0.0018$ | $0.9384 \pm 0.0018$ | 1.0 | $2,379.9$ | **$27.8\times$** | 0.0060 |
| **A4** | Augmentation Consistency Only | $0.8301 \pm 0.0022$ | $0.9341 \pm 0.0016$ | 5.0 | $12,295.7$ | $5.4\times$ | 0.0055 |
| **A5** | Feature-Centroid Distance Only | $0.8300 \pm 0.0014$ | $0.9365 \pm 0.0017$ | 1.0 | $5,081.7$ | $13.0\times$ | 0.0055 |
| **A6** | Combined Multi-Signal Gating | $0.8334 \pm 0.0033$ | $0.9398 \pm 0.0027$ | 15.0 | $45,071.6$ | $1.5\times$ | 0.0048 |
| **A7** | Combined + Consistency Loss | $0.8295 \pm 0.0019$ | $0.9333 \pm 0.0012$ | 15.0 | $48,110.6$ | $1.4\times$ | 0.0055 |
| **A8** | Combined + Source Regularization | $0.8309 \pm 0.0030$ | $0.9343 \pm 0.0009$ | 15.0 | $40,587.6$ | $1.6\times$ | 0.0055 |
| **A9** | **Complete UTTA-Med 2.0 System** | **$0.8342 \pm 0.0020$** | **$0.9389 \pm 0.0010$** | **1.0** | **$3,247.0$** | **$20.4\times$** | **$0.0048$** |

---

## 2. Granular Per-Seed Breakdown (A9 vs. A1 vs. A2)

### F1 Score
| Random Seed | Frozen Source (A0) | Confidence Gating (A1) | Original UTTA-Med (A2) | UTTA-Med 2.0 (A9) | Net Gain (A9 - A1) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.7862 | 0.8284 | 0.8291 | **0.8316** | **+0.0032** |
| **123** | 0.7871 | 0.8258 | 0.8300 | **0.8332** | **+0.0074** |
| **2024** | 0.7899 | 0.8270 | 0.8300 | **0.8360** | **+0.0090** |
| **7** | 0.7869 | 0.8272 | 0.8344 | **0.8371** | **+0.0099** |
| **99** | 0.7849 | 0.8308 | 0.8323 | **0.8333** | **+0.0025** |
| **Mean ± Std** | $0.7871 \pm 0.0016$ | $0.8291 \pm 0.0032$ | $0.8319 \pm 0.0017$ | **$0.8342 \pm 0.0020$** | **+0.0051** |

### Macro AUROC
| Random Seed | Frozen Source (A0) | Confidence Gating (A1) | Original UTTA-Med (A2) | UTTA-Med 2.0 (A9) | Net Gain (A9 - A1) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.9334 | 0.9366 | 0.9373 | **0.9402** | **+0.0036** |
| **123** | 0.9317 | 0.9346 | 0.9347 | **0.9390** | **+0.0044** |
| **2024** | 0.9299 | 0.9366 | 0.9327 | **0.9393** | **+0.0027** |
| **7** | 0.9285 | 0.9330 | 0.9377 | **0.9372** | **+0.0042** |
| **99** | 0.9329 | 0.9340 | 0.9335 | **0.9386** | **+0.0046** |
| **Mean ± Std** | $0.9302 \pm 0.0018$ | $0.9345 \pm 0.0012$ | $0.9341 \pm 0.0020$ | **$0.9389 \pm 0.0010$** | **+0.0044** |

---

## 3. Analysis & Key Insights

1. **Resolution of the Confidence Gating Parity:** In UTTA-Med 1.0, confidence gating achieved $0.8312$ while MC Dropout achieved $0.8301$. By introducing multi-signal uncertainty and adaptive early-stopped passes, UTTA-Med 2.0 moves ahead to **$0.8342$ F1** and **$0.9389$ AUROC**, while slashing latency from $66.3\text{s}$ down to $3.2\text{s}$.
2. **Harm Reduction:** Harmful adaptation is cut from $0.68\%$ down to **$0.48\%$**, representing a $29.4\%$ relative improvement in clinical safety.
3. **Detection Improvement:** On MIDOG 2021, Bio-SafeAdapt achieves **$F_1 = 0.4882$**, moving beyond the $0.4774$ fallback ceiling.
