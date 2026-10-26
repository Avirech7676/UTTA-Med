# Graphical Abstract Specification

**Manuscript Title:** Fail-Closed Test-Time Adaptation and Runtime Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Severe Domain Shift  
**Target Venue:** IEEE Transactions on Medical Imaging (TMI) / Medical Image Analysis (MedIA)

---

### Graphical Abstract Panels & Flow

```
+-------------------------------------------------------------------------------------------------------+
|                                    INPUT STREAM & CHROMATIC DOMAIN SHIFT                              |
|  [ Target WSI Stream: Camelyon17 Hospital 2 & MIDOG Scanner 3 (Severe Stain / Scanner Variation) ]    |
+-------------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+-------------------------------------------------------------------------------------------------------+
|                                    SAFEADAPT RUNTIME SAFEGUARD ENGINE                                  |
|                                                                                                       |
|  [ Branch 1: Classification (ResNet-18) ]           [ Branch 2: Dense Detection (Faster R-CNN) ]      |
|  • Beer-Lambert OD Stain Perturbation:              • Cellularity-Mitosis Ratio (CMR) Gate:           |
|    OD = -log10(I / I0) -> KL Consistency Loss         rho_cell(x) >= 0.15 (H-OD Nuclear Deconvolution)|
|  • Sinkhorn-Knopp Optimal Transport:                • Eliminates 14 Acellular Stroma False Alarms     |
|    min <Q, C> - eps H(Q) s.t. Q^T 1_B = r           • Non-Destructive Partial Recoil Buffer           |
|  • Cosine Source Anchoring & L_inf <= 0.05 Trust    • Unlocks Continuous Stream Adaptation            |
+-------------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+-------------------------------------------------------------------------------------------------------+
|                                   CLINICAL DEPLOYMENT & METRIC OUTCOMES                               |
|                                                                                                       |
|  • Slide 29 Attractor Reversal:   0.8592 -> 0.9324 AUROC (+0.0732 boost on severe outlier)            |
|  • 5-Seed Primary Camelyon17:     0.9495 +- 0.0134 Macro AUROC (p = 0.0312, paired Wilcoxon)         |
|  • Sharp Probability Calibration: ECE cut by 43.2% (0.1281 -> 0.0745)                                 |
|  • MIDOG Scanner 3 Detection:     F1 = 0.5160 +- 0.0050 (Delta F1 = +0.0575 over frozen source)       |
|  • Clinical Hardware Profile:     53.5 ms latency / 32 patches, 474 MB VRAM (Nvidia Tesla T4)        |
+-------------------------------------------------------------------------------------------------------+
```

### Visual Assets for Upload
1. Primary composite banner: `paper/figures/fig_v2_slide29_attractor_reversal.png` (Panel 1) combined with `paper/figures/fig_v2_benchmark_auroc_ece.png` (Panel 2).
2. Diagnostic telemetry proof: `paper/figures/fig_v2_gpu_training_convergence.png`.
3. High-resolution PDF equivalents available in `paper/figures/*.pdf`.
