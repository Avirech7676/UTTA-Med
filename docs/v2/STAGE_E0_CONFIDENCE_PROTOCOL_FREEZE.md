# Stage E0: Confidence-Only SafeAdapt Protocol Freeze

**Auditor:** Independent Scientific Auditor, Senior PyTorch Engineer, and Medical Imaging Reproducibility Researcher  
**Date:** October 2026  
**Status:** Protocol Frozen for Stage E1  
**Target:** Single Frozen Candidate Architecture for Production SafeAdapt  

---

## 1. Executive Summary

In Stage D2, the independent scientific evaluation demonstrated that:
1. MC-dropout predictive variance is near-perfectly colinear with softmax confidence ($\rho = -0.9917$).
2. MC-only selection accepted zero candidate patches that confidence gating rejected.
3. Confidence gating achieved identical AUROC, AUPRC, and F1 at matched budget while running $4.39\times$ faster with zero stochastic forward passes.

Consequently, Stage D2 rendered a **NO-GO** decision for the MC-dropout / Hybrid research track. This document formalizes the complete parameter freeze for the unified **Confidence-Gated SafeAdapt Architecture** to be evaluated under Stage E1.

---

## 2. Complete Frozen Configuration Specification

The following parameters are strictly frozen. No hyperparameter tuning or adjustment on Hospital-2 data is permitted.

```
================================================================================
FROZEN CONFIDENCE-GATED SAFEADAPT ARCHITECTURE (STAGE E0 FREEZE)
================================================================================
1. Source Model & Checkpoint
   - Architecture: ResNet-18 (Torchvision backbone + 2-layer classifier)
   - Checkpoint Path: checkpoints/camelyon17_resnet18_source_s42_BEST.pt
   - Checkpoint Seed: 42
   - Checkpoint SHA-256: f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000
   - Model Immutability: True (reference source weights preserved in immutable deep copy)

2. Parameter Update Scope
   - Updatable Parameters: BatchNorm2d affine parameters ONLY (weight and bias)
   - Excluded Parameters: All convolutional weights, linear layers, classifier heads (requires_grad = False)
   - BatchNorm Running Stats: FROZEN (running_mean and running_var locked; no batch tracking)
   - Module Mode during Adaptation: eval() for Dropout; eval() for BatchNorm running stats

3. Optimization & Loss Formulation
   - Objective: Shannon entropy minimization on selected patches:
       L_batch = (sum_{i in Selected} H(p_i)) / max(1, |Selected|)
       H(p_i) = -[p_i * log(p_i) + (1 - p_i) * log(1 - p_i)]
   - Optimizer: torch.optim.Adam
   - Learning Rate: 1.0e-5 (1e-5)
   - Adam Betas: (0.9, 0.999)
   - Adam Epsilon: 1.0e-8
   - Weight Decay: 0.0
   - Gradient Clipping: max_norm = 1.0 (L2 norm)
   - Batch Size: 32 patches per adaptation step

4. Confidence Selection Policy
   - Primary Operational Threshold: tau_conf = 0.90
   - Selection Rule: Accept patch x_i if max(p_i, 1.0 - p_i) >= tau_conf
   - Zero-Selected Bypass: If |Selected| == 0, skip backward pass and optimizer step entirely

5. Stability Controller & Parameter Bounds
   - Max L_inf Parameter Displacement: Delta_max = 0.05 (|theta_t - theta_0|_inf <= 0.05)
   - Non-Finite Guardrail: Pre-update check rejecting NaN/Inf loss; post-backward check rejecting NaN/Inf grads
   - Atomic Rollback: Restores model parameters, buffers, and optimizer state to last accepted state

6. Collapse Monitor (Observation & Active Interventions)
   - Sliding Window Size: W = 5 batches
   - Minimum Logit Variance: var_min = 0.08
   - Maximum Disagreement with Source: D_max = 0.35
   - Minimum Predictive Entropy: H_min = 0.05
   - Rollback Alarm Trigger: 2 consecutive suspicious batches -> Atomic Rollback
   - Fallback Alarm Trigger: 4 consecutive suspicious batches -> Immutable Source Fallback

7. Execution Semantics
   - Prediction Timing: PRE-ADAPTATION (batch predictions recorded before optimizer step)
   - Reset Policy: Slide-level reset at every WSI boundary (model, optimizer, monitor reset to source)
================================================================================
```

---

## 3. Dissecting the Confidence Threshold: Natural vs. Matched Budget

A critical methodological rule of Stage E0 is to prevent the silent promotion of the matched-budget threshold as a clinically optimal value:

### A. The Natural Threshold ($\tau_{\text{conf}} = 0.90$)
- **Origin & Provenance:** Established in Stage D1 based on standard high-confidence entropy-filtering heuristics in test-time adaptation literature.
- **Empirical Behavior on Hospital 1:**
  - Patch Coverage: **$87.5\%$** (280 / 320 patches).
  - AUROC: **$0.9738$** (improved from $0.9736$ Frozen Source).
  - AUPRC: **$0.9545$** (improved from $0.9542$ Frozen Source).
  - F1 Score: **$0.8596$** (slight boundary shift from $0.8632$ Frozen Source).
  - Latency: $204.88\text{s}$ ($20.5\text{s}$ / batch).
- **Clinical Role:** Represents the natural operating policy where high-confidence tissue adapts the feature representations.

### B. The Matched-Budget Threshold ($\tau_{\text{conf}} = 0.9931$)
- **Origin & Provenance:** Calibrated strictly on Calibration WSIs `[10, 12, 14, 16, 18]` solely to match MC-only's $72.2\%$ coverage for a fair computational benchmark.
- **Empirical Behavior on Hospital 1:**
  - Patch Coverage: **$68.8\%$** (220 / 320 patches).
  - AUROC: **$0.9737$**.
  - AUPRC: **$0.9544$**.
  - F1 Score: **$0.8632$** (fully preserves baseline F1).
  - Latency: $118.09\text{s}$ ($11.8\text{s}$ / batch).
- **Methodological Distinction:**
  > [!WARNING]
  > We do **not** claim $\tau = 0.9931$ is an optimal clinical threshold. It was derived as a synthetic benchmark artifact to match MC-only patch counts. However, its empirical performance reveals that more conservative gating ($68.8\%$ vs $87.5\%$) preserves classification F1 while reducing backprop overhead.

### Protocol Decision for Stage E1
In Stage E1, the primary evaluation shall use the **Natural Threshold ($\tau = 0.90$)**, and report a prespecified sensitivity analysis across $\tau \in \{0.85, 0.90, 0.95, 0.99\}$.

---

## 4. Architectural Boundaries and Non-Access Guarantees

1. **Zero Access to Hospital 2:**
   Hospital-2 test data (Center 2, Slides 20–29) remains completely sealed. Zero images, patches, or annotations have been loaded or evaluated.
2. **Frozen v1 Artifacts Preserved:**
   No v1 checkpoints, random seeds, Table 1/2 results, or manuscript claims have been altered.
3. **No Clinical Deployment Claims:**
   SafeAdapt is evaluated strictly as an algorithmic reliability mechanism. No claims of unconditional clinical safety or statistical non-inferiority are made.
