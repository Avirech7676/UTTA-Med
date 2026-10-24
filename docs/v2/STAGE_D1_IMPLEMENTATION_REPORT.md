# Stage D1: Implementation Report — Faithful and Efficient Gating

**Author:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist  
**Date:** October 2026  
**Status:** Implemented & Verified  
**Package:** `src/v2/gating/`  

---

## 1. Overview & Architecture

Stage D1 implements a clean, modular, and faithful test-time adaptation gating architecture in the isolated `src/v2/gating` namespace, integrating seamlessly with the existing `SafeAdaptController` (Stage B) and `CollapseMonitor` (Stage C2).

```
src/v2/gating/
├── __init__.py           # Public exports
├── policies.py           # Pluggable gating decision policies (A, B, C, D)
├── adapter.py            # GatedSafeAdaptAdapter integrating policies with SafeAdaptController
└── telemetry.py          # Granular patch, update, and compute telemetry tracker
```

---

## 2. Implementation of Four Gating Policies

### Policy A: Frozen Source (`FrozenSourcePolicy`)
- **Mechanism:** Deterministic baseline. Gating keep mask is strictly `False` for all patches ($w_i = 0, \forall i$).
- **Compute:** Zero backward passes, zero optimizer updates, zero stochastic passes. Predictions are evaluated directly through the frozen source model.

### Policy B: Confidence-Only (`ConfidenceGatingPolicy`)
- **Mechanism:** Deterministic patch-level screening.
- **Formulation:** Evaluates $p_i = \sigma(z_i)$ and confidence $c_i = \max(p_i, 1 - p_i)$. Selects patch $i$ if $c_i > \tau_{\mathrm{conf}}$ (default $\tau_{\mathrm{conf}} = 0.90$).
- **Compute:** 1 deterministic forward pass; 0 stochastic MC passes.

### Policy C: MC-Only (`MCOnlyGatingPolicy`)
- **Mechanism:** Faithful reproduction of v1 UTTA-Med 20-pass MC-Dropout predictive variance gating.
- **Formulation:** Sets `model.eval()` while enabling `dropout.train()`. Executes $K = 20$ stochastic forward passes without mutating BatchNorm running statistics. Computes sample variance across probabilities:
  $$U_i = \text{Var}_{k=1 \dots 20}(p_{i,k}) \quad (\text{unbiased}=\text{False}).$$
  Selects patch $i$ if $U_i < \tau_{\mathrm{unc}}$ ($\tau_{42} = 8.875383 \times 10^{-6}$ from `configs/tau_by_seed.yaml`).
- **Compute:** $K = 20$ stochastic forward passes per patch.

### Policy D: Efficient Hybrid Gating (`HybridGatingPolicy`)
- **Mechanism:** Two-stage decision arbitration that preserves epistemic uncertainty filtering while eliminating redundant MC passes on unambiguous patches.
- **Stage 1 (Deterministic Screening):**
  Evaluates $c_i = \max(p_i, 1 - p_i)$ via a single forward pass:
  1. *Unambiguously Confident ($c_i \ge \theta_{\mathrm{high}}$, default 0.95):* Auto-accepted directly without MC passes ($w_i = 1$).
  2. *Low-Confidence Noise ($c_i < \theta_{\mathrm{low}}$, default 0.65):* Auto-rejected directly without MC passes ($w_i = 0$).
- **Stage 2 (Selective MC-Dropout):**
  For the ambiguous margin $\theta_{\mathrm{low}} \le c_i < \theta_{\mathrm{high}}$:
  1. Extracts ambiguous patches $x_{\mathrm{amb}} = x[\text{ambiguous}]$.
  2. Executes $K = 20$ stochastic MC-dropout passes **exclusively on $x_{\mathrm{amb}}$**.
  3. Computes predictive variance $U_{\mathrm{amb}}$. Accepts ambiguous patch $i$ if $U_i < \tau_{\mathrm{unc}}$.
- **Efficiency Gain:** If 80% of patches are confidently separated, MC-dropout is executed on only 20% of patches (or 0 passes for clean batches), slashing FLOPs and latency by 70–90% while retaining uncertainty protection on difficult tissue.

---

## 3. Integration with SafeAdapt Controller

The `GatedSafeAdaptAdapter` couples gating decisions with the SafeAdapt Stage B controller:

1. **Masked Entropy Loss Formulation:**
   When $\sum w_i > 0$, the loss is strictly evaluated over selected patches:
   $$\mathcal{L} = \frac{\sum_{i=1}^B w_i \cdot H(p_i)}{\sum_{i=1}^B w_i}$$
2. **Zero-Selection Graceful Bypass:**
   When $\sum w_i = 0$ (all patches rejected by the gate), the adapter completely bypasses the backward pass and optimizer step. Predictions are returned in `model.eval()` mode with zero parameter updates.
3. **SafeAdapt Controller Guarantees:**
   Every candidate update undergoes:
   - Finite loss, gradient, parameter, and prediction validation.
   - Independent gradient clipping ($\|\mathbf{g}\|_2 \le 1.0$).
   - Parameter displacement projection ($\|\theta - \theta_0\|_\infty \le 0.20$).
   - Slide-boundary isolation and atomic pre-update rollback on failure.
   - Collapse monitoring in observation mode (`MonitorMode.OBSERVATION`).

---

## 4. Telemetry Tracker Architecture

The `GatingTelemetryTracker` records fine-grained per-batch metrics:
- **`eligible_patches`**: Total patches presented in batch.
- **`selected_patches`**: Patches meeting the gating criterion.
- **`routed_to_mc_patches`**: Patches evaluated via MC-dropout.
- **`mc_passes_executed`**: Stochastic passes executed ($0$ or $20$).
- **`patch_coverage`**: $\sum N_{\mathrm{selected}} / \sum N_{\mathrm{eligible}}$.
- **`update_coverage`**: $\sum N_{\mathrm{accepted\_updates}} / \sum N_{\mathrm{attempted\_updates}}$.
- **`prediction_coverage`**: Exactly $100\%$ (every patch receives a valid prediction).
- **`mc_compute_ratio`**: Actual MC passes / theoretical full MC passes ($20 \times B_{\mathrm{total}}$).
- **`gating_latency_ms`**, **`step_latency_ms`**, **`total_latency_ms`**.
