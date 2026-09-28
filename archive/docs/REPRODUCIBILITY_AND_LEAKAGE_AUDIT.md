# UTTA-Med: Scientific Reproducibility & Leakage Audit

This document records the formal protocol audit conducted across all 5 seeds ($\{42, 123, 2024, 7, 99\}$) and ablation experiments on the **Camelyon17-WILDS** benchmark.

---

## 1. Information Flow & Split Boundary Protocol

```
Source Domain (Centers {0, 3, 4})
  │  (Supervised ERM training: 302,436 patches, 30 WSIs)
  ▼
OOD Validation Domain (Center {1})
  │  (34,904 patches, 10 WSIs)
  │  • Early stopping selection of BEST checkpoint
  │  • Predictive uncertainty U(x) calculation (N=20 passes)
  │  • Gating threshold tau selected at 70th percentile of U(x)
  │  • Matched-coverage threshold calibration for Confidence & Entropy gates
  │  • Hyperparameters (LR=1e-5, steps=1, e_margin=0.4) FROZEN
  ▼
Target Domain (Center {2})
     (85,054 patches, 10 WSIs)
     • Unlabeled test stream adaptation (TTA)
     • Labels are strictly withheld during adaptation
     • Target labels accessed ONLY once after streaming completes for evaluation
```

---

## 2. Checkpoint-by-Checkpoint Audit Checklist

| Audit Rule | Compliance | Verification Evidence |
| :--- | :---: | :--- |
| **1. No Hospital-2 labels used for $\tau$** | **VERIFIED (PASS)** | $\tau$ is calibrated strictly at the 70th percentile of predictive variance on Center 1 validation data (`docs/dataset_protocol.md`, `configs/camelyon17.yaml`). |
| **2. No Hospital-2 labels used to choose TTA LR** | **VERIFIED (PASS)** | Learning rate $\eta = 10^{-5}$ was selected via Center 1 validation sweep prior to target test streaming (`results/metrics/camelyon17_resnet18_tent_eata_s42.json`). |
| **3. No Hospital-2 labels used to choose MC $N$** | **VERIFIED (PASS)** | $N_{\text{MC}} = 20$ was chosen on Center 1 validation correlation ($r = 0.425$ at $N=20$ vs $0.437$ at $N=50$) before running test evaluation. |
| **4. No Hospital-2 labels used to select adaptation steps** | **VERIFIED (PASS)** | Gradient steps per batch $k = 1$ is frozen. Validation showed $k \ge 5$ triggers logit drift; test labels were never consulted. |
| **5. No Hospital-2 labels used to select final method** | **VERIFIED (PASS)** | Baselines (Source, Tent, EATA, Random, Confidence, Entropy, UTTA) were all preregistered and run under identical streaming conditions. |
| **6. No seed-specific test tuning** | **VERIFIED (PASS)** | Protocol hyperparameters ($\tau$, $\eta$, $N$, BN freeze mode) are completely uniform across all 5 seeds. Seeds 7 and 99 were neither re-tuned nor discarded. |
| **7. No Grad-CAM cherry-picking** | **VERIFIED (PASS)** | Grad-CAM samples were selected objectively based on automated prediction/ground-truth confusion matrices and uncertainty quantiles recorded in `gradcam_manifest.json`. |

---

## 3. Boundary Integrity Verification

1. **WSI Overlap Check**:
   - $\text{WSIs}_{\text{train}} \cap \text{WSIs}_{\text{val}} = \emptyset$ (30 vs 10 distinct WSIs).
   - $\text{WSIs}_{\text{train}} \cap \text{WSIs}_{\text{test}} = \emptyset$ (30 vs 10 distinct WSIs).
   - $\text{WSIs}_{\text{val}} \cap \text{WSIs}_{\text{test}} = \emptyset$ (Hospital 1 vs Hospital 2).
   - Confirmed by unit test: [`tests/test_no_leakage.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/tests/test_no_leakage.py) (PASSED).

2. **Model Parameter Separation**:
   - Adaptation affects only scale ($\gamma$) and shift ($\beta$) parameters of `BatchNorm2d`.
   - Running mean and running variance are frozen (`bn_freeze_stats`).
   - Feature weights (convolutions) and classifier head remain strictly frozen during TTA.
