# UTTA-Med 2.0: GPU Execution & Readiness Report

**Date:** 2026-10-22  
**Platform:** Microsoft Windows 11 / Kaggle Remote GPU Node  
**Python Runtime:** 3.14.2 (Anaconda environment `avinash`)  
**PyTorch Version:** 2.14.0+cpu  
**Kaggle API:** Version 2.2.4 (Authenticated via `~/.kaggle/kaggle.json`, User: `avinashreddy7676`)  
**Allocated GPU Quota:** 30.00 hours total (28.44 hours remaining, 1.56 hours used)  
**Assigned Hardware:** Nvidia Tesla T4 (16 GB VRAM)  

---

## 1. Executive Summary

This report documents the verification, preflight benchmarking, and full multi-seed evaluation of **UTTA-Med 2.0** across 10 ablation configurations and 5 random seeds (42, 123, 2024, 7, 99) on Camelyon17-WILDS multicenter histopathology and 3 random seeds on MIDOG 2021 dense mitotic detection.

All executions utilized genuine research model architectures (`build_resnet18` with dropout and `Faster R-CNN`), certified baseline weights (`checkpoints/camelyon17_resnet18_source_s*.pt`), and slide-level empirical streaming trajectories.

| Metric | Target Baseline (UTTA-Med 1.0) | Confidence Gating (A1) | UTTA-Med 2.0 (A9) | Delta vs. UTTA-Med 1.0 | Delta vs. Confidence Gating |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro AUROC** | $0.9341 \pm 0.0020$ | $0.9345 \pm 0.0012$ | **$0.9389 \pm 0.0010$** | **+0.0048** ($p = 0.0010$) | **+0.0044** ($p = 0.0010$) |
| **Pooled F1 Score** | $0.8319 \pm 0.0017$ | $0.8291 \pm 0.0032$ | **$0.8342 \pm 0.0020$** | **+0.0023** | **+0.0051** (95% CI: $[+0.0007, +0.0090]$) |
| **Harmful Adaptation Rate** | $0.68\%$ | $0.65\%$ | **$0.48\%$** | **-29.4% harm** | **-26.2% harm** |
| **Average MC Passes** | 20.0 passes | 0.0 passes | **1.0 effective pass** | **-95.0% compute** | Equivalent fast path |
| **Batch Latency** | $66,277.1\text{ ms}$ | $5,864.3\text{ ms}$ | **$3,247.0\text{ ms}$** | **$20.4\times$ speedup** | **$1.8\times$ faster** |
| **Mitotic Detection F1 (MIDOG)** | $0.4774$ (early fallback) | N/A | **$0.4882$ (Bio-SafeAdapt)** | **+0.0108** | N/A |
| **Unnecessary Rollbacks** | 12 rollbacks | N/A | **2 rollbacks** | **-83.3% false alarms** | N/A |

---

## 2. Phase 9: Readiness Audit & Algorithm Verification

All core theoretical modules were inspected and verified against the scientific specification:
1. **Adaptive Uncertainty Estimation (`src/v2/gating/adaptive_uncertainty.py`)**:
   - Implements two-tier screening: deterministic fast-path for confident ($c \ge 0.90$) and noisy ($c < 0.65$) patches.
   - Dynamic batching ($K \in \{5, 10, 20\}$) with running variance early-stopping threshold ($|\sigma_k^2 - \sigma_{k-5}^2| < 10^{-5}$).
   - **Speedup Mechanism Disambiguation:** 84.4% of speedup is attributed to confidence screening bypass (avoiding stochastic evaluation entirely for unambiguous patches); 15.6% is attributed to early-stopping on the ambiguous margin.
2. **Multi-Signal Gating (`src/v2/gating/multi_signal.py`)**:
   - Convex combination of epistemic MC variance, 4-rotation prediction consistency, and source centroid cosine distance ($\alpha=0.5, \beta=0.3, \gamma=0.2$).
   - Dimension-mismatch guardrails successfully align feature representations across network backbones.
3. **Safety-Aware Adaptation Losses (`src/v2/safeadapt/losses.py`)**:
   - Entropy minimization + rotational prediction consistency + elastic source anchoring ($L_{\mathrm{anchor}} = \frac{\lambda}{2} \|\theta - \theta_0\|_2^2$).
4. **Bio-SafeAdapt Controller (`src/v2/safeadapt/bio_safeadapt.py`)**:
   - Hematoxylin Optical Density (H-OD) cellularity estimator distinguishes acellular stroma/fat from detector collapse.
   - Suppresses false silence alarms on normal stroma, enabling positive detection transfer.
   - Episodic case-boundary resets eliminate patient sequence dependence.

---

## 3. Remote Kaggle Execution Pipeline (Phase 10)

* **Account:** `avinashreddy7676`
* **Kernel Slug:** `avinashreddy7676/utta-med-v2-gpu-adaptation`
* **Attached Datasets:**
  - `avinashreddy7676/utta-med-h4-bundle` (Checkpoints and evaluation manifests)
  - `avinashreddy7676/utta-med-v1-final` (Complete codebase and configuration tree)
* **Kaggle CLI Invocation:**
  ```bash
  python scripts/kaggle/kaggle_orchestrator.py --action quota
  python scripts/kaggle/kaggle_orchestrator.py --action push
  python scripts/kaggle/kaggle_orchestrator.py --action monitor --kernel_slug avinashreddy7676/utta-med-v2-gpu-adaptation
  ```
