# Empirical Verdict on Model Performance Improvement

**Project:** UTTA-Med Project 2 — Scientific Verdict  
**Date:** October 24, 2026  

---

## 1. Scientific Verdict Summary

1. **Historical ResNet-18 Benchmark:**
   * Confidence Gating ($M1$, $F_1 = 0.8310 \pm 0.0321$, $\text{AUROC} = 0.9355 \pm 0.0218$) is the verified SOTA on ResNet-18.
   * Original UTTA-Med ($M2$, $F_1 = 0.8296 \pm 0.0325$) does **not** statistically outperform Confidence Gating ($\Delta F_1 = -0.0014, t = -0.76, df = 4, p = 0.4850$).
2. **DenseNet-121 Architectural Scaling:**
   * The previously reported DenseNet-121 $F_1 = 0.8723$ was an analytical projection from a 128-sample CPU test and has been rejected.
   * Real GPU training is actively underway on Kaggle (Seed 42 Epoch 1).
   * In strict accordance with the Zero Fabrication policy, empirical performance metrics will be published exclusively upon completion of real GPU evaluations.
