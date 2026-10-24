# SafeAdapt Technical Specification: Safety-Guaranteed Test-Time Adaptation

**Document Version:** 2.0-Design  
**Date:** October 10, 2026  
**Auditor / Architect:** Independent Medical Imaging AI Auditor & Machine Learning Scientist  
**Target System:** UTTA-Med v2 (SafeAdapt Extension)  
**Safety Invariant:** *The adapted model's performance on any target clinical whole-slide image must be mathematically and empirically guaranteed not to catastrophically degrade below the frozen source model.*

---

## 1. Architectural Architecture & Core Mechanisms

SafeAdapt introduces five modular safety layers operating strictly at inference time without requiring target ground truth labels:

```
[ Incoming Target Batch x ]
            │
            ▼
┌───────────────────────────────────────────────┐
│  Layer 1: Hybrid Selective Gate               │
│  - Fast Deterministic Confidence Filter       │
│  - Conditional Epistemic Variance Arbiter    │
└───────────────────────┬───────────────────────┘
                        │ Accepted subset w(x) > 0
                        ▼
┌───────────────────────────────────────────────┐
│  Layer 2: Parameter-Space Safeguards          │
│  - Gradient L2 Norm Clipping (max_norm = 1.0) │
│  - Source Anchor Distance Constraint          │
│    ||θ_TTA - θ_source||_∞ ≤ ε_max             │
└───────────────────────┬───────────────────────┘
                        │ Proposed Update Δθ
                        ▼
┌───────────────────────────────────────────────┐
│  Layer 3: Online Prediction-Collapse Alarm    │
│  - Rolling Window Predicted Prevalence p̂_win │
│  - Logit Variance & Mean Drift Monitor        │
└───────────────────────┬───────────────────────┘
                        │
            ┌───────────┴───────────┐
            │ Alarm Triggered?      │
           YES                      NO
            │                       │
            ▼                       ▼
┌───────────────────────┐ ┌───────────────────────┐
│ Layer 4: Rollback     │ │ Apply Parameter Step  │
│ - Revert to Checkpoint│ │ Commit θ_TTA ← θ + Δθ │
│ - Freeze TTA Engine   │ │ Update Buffer         │
│ - Fallback to Source  │ └───────────────────────┘
└───────────────────────┘
```

---

## 2. Specification of Safety Mechanisms

### 2.1 Mechanism 1: Frozen-Source Fallback (Shadow Model Engine)
- **Objective:** Ensure zero clinical risk of deploying degraded predictions.
- **Specification:**
  - Maintain an immutable, in-memory instance of the pre-trained source model $\mathcal{M}_{\text{source}}$ alongside the adapting instance $\mathcal{M}_{\text{adapted}}$.
  - Both models process the input batch. By default, $\mathcal{M}_{\text{adapted}}$ produces diagnostic predictions.
  - If the collapse alarm trips or adaptation fails sanity bounds, the system automatically falls back to outputting $\mathcal{M}_{\text{source}}(x)$ for the current batch and all subsequent batches.
- **Label-Free Feasibility:** Fully label-free. Requires only retaining initial weights in GPU/CPU memory ($\sim 44\text{ MB}$ for ResNet-18, $\sim 98\text{ MB}$ for ResNet-50).

---

### 2.2 Mechanism 2: Online Prediction-Collapse Alarm
- **Objective:** Detect single-class negative attractor drift ($p \to 0$) or single-class positive saturation ($p \to 1$) in real time without target ground truth.
- **Mathematical Formulation:**
  - Maintain an exponential moving average (EMA) of batch predicted positive rates:
    $$\hat{\pi}_t = \alpha \hat{\pi}_{t-1} + (1 - \alpha) \frac{1}{B} \sum_{i=1}^B \mathbb{I}(\sigma(z_i) \ge 0.5), \quad \alpha = 0.95$$
  - Maintain running mean and variance of raw output logits:
    $$\mu_{z, t} = \alpha \mu_{z, t-1} + (1 - \alpha) \bar{z}_t, \quad \sigma_{z, t}^2 = \alpha \sigma_{z, t-1}^2 + (1 - \alpha) \text{Var}(z_t)$$
  - **Alarm Trigger Conditions:**
    1. **Prevalence Annihilation:** $\hat{\pi}_t < \pi_{\min}$ (e.g., $\pi_{\min} = 0.01$ over a window of $W = 500$ patches).
    2. **Extreme Logit Drift:** $|\mu_{z, t} - \mu_{z, \text{source}}| > \Delta z_{\max}$ (e.g., $\Delta z_{\max} = 3.0$).
    3. **Logit Variance Collapse:** $\sigma_{z, t}^2 < \epsilon_{\text{var}}$ (e.g., $\sigma_{z, t}^2 < 0.05$, indicating constant output).
- **Label-Free Feasibility:** Fully label-free. Computes statistics solely on predicted probabilities and logits.

---

### 2.3 Mechanism 3: Gradient and Parameter-Update Safeguards
- **Objective:** Prevent numerical explosion (as seen in ResNet-50 NaNs) and bound cumulative representation drift.
- **Mathematical Formulation:**
  1. **Gradient Clipping:** Before optimizer step:
     $$g \leftarrow g \cdot \min\left(1, \frac{M_{\text{clip}}}{\|g\|_2}\right), \quad M_{\text{clip}} = 1.0$$
  2. **Bounded Parameter Box ($\ell_\infty$ Projection):** After optimizer step, project BatchNorm affine parameters back into an $\epsilon$-ball around initial source weights:
     $$\gamma_l \leftarrow \text{clamp}(\gamma_l, \gamma_{l, \text{source}} - \epsilon_\gamma, \gamma_{l, \text{source}} + \epsilon_\gamma)$$
     $$\beta_l \leftarrow \text{clamp}(\beta_l, \beta_{l, \text{source}} - \epsilon_\beta, \beta_{l, \text{source}} + \epsilon_\beta)$$
     where $\epsilon_\gamma = 0.20$ and $\epsilon_\beta = 0.20$.
  3. **Elastic Weight Anchor ($L_2$ Regularization):** Augment adaptation loss with an anchor penalty:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{entropy}} + \lambda_{\text{anchor}} \sum_{l} (\|\gamma_l - \gamma_{l, \text{source}}\|_2^2 + \|\beta_l - \beta_{l, \text{source}}\|_2^2)$$
     with $\lambda_{\text{anchor}} = 0.01$.
- **Label-Free Feasibility:** Fully label-free. Operates strictly within the optimizer update step.

---

### 2.4 Mechanism 4: Label-Free Rolling Checkpoint Rollback
- **Objective:** Recover previous healthy model states when sudden slide-level shift causes gradient contamination.
- **Specification:**
  - Maintain a rolling FIFO checkpoint ring buffer of the last $K$ parameter states:
    $$\mathcal{B} = \{\theta_{t - K\Delta}, \dots, \theta_{t - \Delta}, \theta_t\}, \quad K = 3$$
  - If a collapse alarm triggers at step $t$:
    1. Immediately roll back model weights to $\theta_{t - K\Delta}$ (the state prior to entering the hazardous tissue zone).
    2. Freeze parameter updates for the remainder of the current Whole-Slide Image (or for a cooldown period of $N_{\text{cooldown}} = 1,000$ patches).
    3. Route predictions through the frozen source model if post-rollback validation statistics fail.
- **Label-Free Feasibility:** Fully label-free. Triggered exclusively by Layer 3 alarm metrics.

---

### 2.5 Mechanism 5: Confidence-vs-Uncertainty Complementarity (Fast-Gate Architecture)
- **Objective:** Eliminate the $20\times$ computational waste of unconditional MC-Dropout while preserving epistemic filtering where it actually matters.
- **Specification:**
  - **Tier 1 (Fast Deterministic Filter):** Compute single-pass softmax confidence $C(x) = \max(p, 1 - p)$.
    - If $C(x) \ge \tau_{\text{high}}$ (e.g., $0.95$): Sample is definitely reliable. Accept for adaptation without MC-Dropout ($1\times$ compute).
    - If $C(x) \le \tau_{\text{low}}$ (e.g., $0.60$): Sample sits near the decision boundary. Definitely ambiguous / high entropy. Reject immediately without MC-Dropout ($1\times$ compute).
  - **Tier 2 (Conditional Epistemic Arbiter):** Only for intermediate patches ($\tau_{\text{low}} < C(x) < \tau_{\text{high}}$, typically $< 15\%$ of patches):
    - Run $N_{\text{MC}} = 10$ stochastic forward passes to compute epistemic variance $U(x)$.
    - Accept if $U(x) < \tau_U$.
  - **Computational Gain:** Reduces average inference compute from $20\times$ down to $\sim 1.8\times$ baseline forward pass, achieving $11\times$ speedup over v1 UTTA-Med.
- **Label-Free Feasibility:** Fully label-free. Evaluated purely on model outputs.

---

### 2.6 Mechanism 6: Slide-Aware Batch Grouping and Calibration
- **Objective:** Eliminate localized slide collapse (e.g., Slide 29 and Slide 28) driven by continuous streaming of thousands of correlated patches from a single tumor-dense or necrosis-dense resection.
- **Specification:**
  - In clinical pathology pipelines, Whole-Slide Image (WSI) metadata is available at inference time (patches from the same slide share a slide identifier, even though labels are absent).
  - **Slide-Wise Episodic Reset:** Reset adaptation parameters $\theta \leftarrow \theta_{\text{source}}$ between distinct Whole-Slide Images, or limit cumulative adaptation steps per slide to $N_{\text{slide\_max}} = 500$ steps.
  - **Slide Tumor Burden Monitoring:** Continuously estimate mean slide tumor fraction $\bar{p}_{\text{slide}} = \frac{1}{N} \sum_{i=1}^N p_i$. Compare against the source model's calibrated baseline to prevent drift.
- **Label-Free Feasibility:** Fully label-free. Utilizes only slide boundary metadata inherent to digital slide scanners.

---

## 3. Development Validation Protocol (Zero Test-Set Leakage)

All thresholds, hyperparameter constants ($\epsilon_\gamma, M_{\text{clip}}, \lambda_{\text{anchor}}, \pi_{\min}, \tau_{\text{high}}, \tau_{\text{low}}$), and alarm sensitivities **must be tuned and validated strictly on Center 1 (OOD Validation Hospital)**:

1. **Synthetic Failure Injection on Center 1:**
   - Synthetically inject adversarially crafted high-entropy batches or corruptions (e.g., Gaussian noise severity 3, severe brightness offsets) into Center 1.
   - Verify that the Prediction-Collapse Alarm trips within $\le 5$ batches of collapse onset.
   - Verify that the Rollback and Frozen-Source Fallback mechanism successfully maintains Center 1 AUROC $\ge 0.949$ (the frozen source baseline).
2. **ResNet-50 Recovery Verification on Center 1:**
   - Run SafeAdapt ResNet-50 on Center 1.
   - Verify that gradient clipping and $\ell_\infty$ parameter bounding eliminate all `NaN` occurrences.
   - Confirm that Center 1 $F_1$ exceeds $0.80$ and AUROC exceeds $0.93$.
3. **Target Isolation Rule:**
   - Under no circumstances will Hospital 2 (Center 2) test patches or labels be loaded during threshold calibration.
