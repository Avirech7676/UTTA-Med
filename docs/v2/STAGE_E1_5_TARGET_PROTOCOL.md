# Stage E1.5: Prespecified Hospital-2 Target Evaluation Protocol

**Lead Investigator & Auditor:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Prespecified Protocol Freeze (Locked Prior to Hospital-2 Access)  
**Target Domain:** Camelyon17-WILDS Center 2 (Hospital 2 Unseen Target Cohort)  
**Target Slide Set:** Test WSIs `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]` ($N_{\text{WSI}} = 10$)  
**Source Checkpoint:** `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`  
**Checkpoint Hash:** `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` (SHA-256)

---

## 1. Executive Protocol Summary

This document establishes the prespecified evaluation protocol for validating UTTA-Med v2 SafeAdapt on the out-of-distribution Hospital-2 target domain. To prevent scientific bias, p-hacking, or post-hoc parameter optimization:
1. All hyperparameters, model architectures, checkpoint weights, and decision rules are locked before unsealing Hospital 2.
2. No post-hoc tuning of the confidence gating threshold ($\tau = 0.90$), learning rate ($\eta = 10^{-5}$), or collapse monitor boundaries will be permitted based on Hospital-2 results.
3. Prior exposure to Hospital-2 under the discarded v1 codebase is formally disclosed.

---

## 2. Mandatory Prior-Exposure Disclosure

> [!WARNING]
> **Prior v1 Hospital-2 Exposure Disclosure:**  
> During the preliminary UTTA-Med v1 study, the Hospital-2 test split was accessed to produce the frozen v1 baseline tables. That study observed severe degradation in ResNet-50 (AUROC falling below 0.50) and gradient instability.  
> **Isolation in v2 Development:**  
> Throughout Stages B, C1, C2, C2.5, D1, D2, E0, and E1, Hospital-2 data and labels have been strictly quarantined. All v2 architectural components (SafeAdapt controller, rollback mechanisms, and collapse monitors) were engineered and calibrated exclusively on Hospital 1. Stage E2 represents a true unblinded evaluation of the frozen v2 system on the target domain.

---

## 3. Prespecified Experimental Arms

The Stage E2 benchmark evaluates four strictly matched experimental arms across all 10 Hospital-2 test slides:

| Arm ID | Nomenclature | Architectural Description | Safety Guarantees |
| :---: | :--- | :--- | :--- |
| **Arm A** | **Frozen Source** | Checkpoint weights $\theta_0$ remain completely static. Zero optimizer steps, zero updates. Model in `eval()` mode. | Baseline anchor (no drift, no adaptation). |
| **Arm B** | **Confidence TTA** | Standard test-time entropy minimization ($\tau=0.90$, Adam $\text{lr}=10^{-5}$, BatchNorm affine only). | None. Vulnerable to runaway gradient drift and collapse. |
| **Arm C** | **Stage B SafeAdapt** | Arm B + numerical finite checks + $L_\infty(\theta, \theta_0) \le 0.05$ parameter displacement bounds + pre-step atomic rollback. | Parameter bounded, finite gradient guaranteed. |
| **Arm D** | **Stage C2 Active SafeAdapt** | Arm C + active rolling collapse monitoring ($W=50, \min=20$) + automatic emergency fallback to $\theta_0$. | Complete dual protection: parameter bounding + label-free behavioral monitoring. |

---

## 4. Cohort Inclusion, Denominator, and Deterministic Ordering

### A. Target Whole-Slide Images
The evaluation includes all 10 Whole-Slide Images comprising the official Camelyon17 test split from Center 2:
$$\mathcal{S}_{\text{target}} = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]$$

### B. Patch Sampling Budget and Computational Feasibility
- Center 2 contains 85,054 total patches in the parquet index.
- A full evaluation of all 85,054 patches across four arms on CPU would require approximately **29 CPU-hours**, exceeding reasonable interactive execution boundaries.
- **Prespecified Protocol:** To balance statistical power with computational feasibility, evaluation evaluates a deterministic uniform spatial stream of **128 patches per WSI** ($N_{\text{total}} = 1,280$ patches across the 10 WSIs).
- **Deterministic Sampling:** Patches are sampled using a fixed spatial stride $s = \lfloor N_{\text{slide\_total}} / 128 \rfloor$, starting at index 0.
- **Batching:** 128 patches per WSI organized into **4 batches of 32 patches** ($B=4$ steps per slide).
- **Adequacy for Collapse Monitoring:** With 128 patches per slide, the monitor easily surpasses its activation threshold ($\min=20$) on Batch 1 (32 patches), and maintains a full sliding window ($W=50$) across Batches 2, 3, and 4, providing multiple sequential opportunities for multi-window escalation ($c \ge 2, c \ge 4$).

---

## 5. Frozen Execution Invariants

1. **Pre-Adaptation Prediction Timing:**  
   Predictions on incoming batch $x_t$ are evaluated and recorded *before* loss calculation and parameter updates:
   $$p_{\text{eval}}(x_t) = \sigma(f_{\theta_t}(x_t)), \quad \text{then} \quad \theta_{t+1} = \text{ControllerStep}(\theta_t, x_t)$$
2. **Atomic Per-WSI Reset:**  
   At the transition between whole-slide images, model parameters are reset to $\theta_0$, the Adam optimizer state is wiped, and the collapse monitor buffer is cleared.
3. **Strict Label Quarantining:**  
   Binary tumor labels $y$ are completely isolated from all adaptation and monitoring loops. Labels are loaded solely by the offline evaluation reporter.

---

## 6. Primary and Secondary Evaluation Metrics

### Primary Endpoints:
1. **Macro Valid AUROC:** Unweighted mean of AUROC scores across all dual-class slides in Hospital 2. (Single-class slides with $N_{\text{tumor}} = 0$ will report `null` / `UNDEFINED` and be excluded from macro AUROC).
2. **Pooled / Overall F1 Score:** Harmonic mean of precision and recall at standard clinical cutoff $0.50$ across all 1,280 patches.
3. **Expected Calibration Error (ECE):** 10-bin equal-width calibration error.
4. **Brier Score:** Mean squared probability error $\frac{1}{N}\sum (p_i - y_i)^2$.

### Secondary & Telemetry Endpoints:
1. **Safety Interventions:** Total counts of non-finite rejections, parameter bound clamps, collapse warnings, atomic rollbacks, and emergency source fallbacks.
2. **Cumulative Displacement:** Maximum observed parameter displacement $\|\theta - \theta_0\|_\infty$.
3. **Execution Latency:** Wall-clock time and latency per patch across all arms.

---

## 7. Reconciled Configuration Specification (`configs/v2/safeadapt_e2_frozen.yaml`)

```yaml
metadata:
  stage: STAGE_E2_TARGET_EVALUATION
  dataset: Camelyon17-WILDS
  target_hospital: Hospital 2 (Center 2)
  status: FROZEN_PRESPECIFIED
  date: "2026-10-11"

model:
  architecture: resnet18
  checkpoint_path: checkpoints/camelyon17_resnet18_source_s42_BEST.pt
  checkpoint_sha256: f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000

adaptation:
  scope: bn_affine_only
  train_bn_running_stats: false
  eval_dropout: true
  batch_size: 32
  loss_function: shannon_entropy
  optimizer:
    name: adam
    lr: 1.0e-05
    betas: [0.9, 0.999]
    eps: 1.0e-08
    weight_decay: 0.0
  grad_clip_norm: 1.0

gating:
  policy: confidence_only
  tau_conf: 0.90
  zero_selected_bypass: true

controller:
  max_param_displacement_linf: 0.05
  reject_non_finite_loss: true
  reject_non_finite_grad: true
  atomic_rollback_on_rejection: true

monitor:
  mode: ACTIVE
  window_size: 50
  min_samples_for_alarm: 20
  min_window_samples: 20
  decision_threshold: 0.5
  min_logit_variance: 0.08
  max_disagreement_rate: 0.35
  min_predictive_entropy: 0.05
  consecutive_suspicious_for_rollback: 2
  consecutive_suspicious_for_fallback: 4

execution:
  prediction_timing: pre_adaptation
  reset_policy: per_wsi_boundary
  eval_slides: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
  patches_per_slide: 128
```

---

## 8. Failure-Handling and Negative-Result Commitments

1. **No Tuning on Failure:** If SafeAdapt fails to improve performance over Frozen Source on Hospital 2, or if it triggers unexpected fallbacks, the result will be reported faithfully as a negative finding.
2. **Single-Run Policy:** Hospital 2 will be executed exactly once under this protocol. No repeated runs with altered parameters will be conducted.
