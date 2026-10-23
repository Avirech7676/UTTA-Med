# Minimum Viable SafeAdapt Controller Specification (Stage B)

**Document Version:** 2.0-MVP  
**Date:** October 10, 2026  
**Status:** Architecture Specification (Pre-Implementation)  
**Governing Principle:** *Implement the minimal, verifiable numerical and parameter safeguards first. Exclude complex heuristic collapse monitors and selective MC-dropout arbiters until core update stability is proven.*

---

## 1. Minimal Viable Scope

The Minimum Viable SafeAdapt (MV-SafeAdapt) controller consists strictly of four testable mechanisms:

1. **Atomic Update Validation:** Rejects any candidate parameter step if the loss, gradients, or post-step weights contain non-finite values (`NaN`, `inf`).
2. **Bounded Parameter Updates:** Constrains BatchNorm affine parameters within a candidate $\ell_\infty$ displacement bound around initial source weights ($\|\theta_t - \theta_0\|_\infty \le \epsilon$).
3. **Immutable Source Fallback:** Maintains a detached, immutable copy of the pre-trained source model state dict that adaptation can never overwrite, allowing instantaneous recovery.
4. **Structured Event Telemetry:** Records every adaptation event (accepted step, rejected step, projection clamp, source fallback invocation) with precise batch metadata.

*Note: Prediction-collapse monitoring (Stage C) and hybrid confidence/MC gating (Stage D) are explicitly deferred until Stage B acceptance tests pass.*

---

## 2. Mathematical and Algorithmic Specification

### 2.1 Mechanism 1: Atomic Update Validation
For incoming batch $B_t$:
1. Compute candidate loss $\mathcal{L}_t$.
   - **Check 1:** If $\mathcal{L}_t$ is non-finite (`torch.isnan(loss) or torch.isinf(loss)`):
     - Abort backward pass.
     - Emit event `REJECT_NONFINITE_LOSS`.
     - Skip update for batch $B_t$.
2. Execute backward pass: $\nabla_\theta \mathcal{L}_t$.
   - **Check 2:** For each adaptable parameter $p$:
     - If `torch.isnan(p.grad).any() or torch.isinf(p.grad).any()`:
       - Zero all gradients: `optimizer.zero_grad(set_to_none=True)`.
       - Emit event `REJECT_NONFINITE_GRAD`.
       - Skip update for batch $B_t$.
3. Execute proposed optimizer step: $\theta_{\text{cand}} \leftarrow \theta_t - \Delta\theta_t$.
   - **Check 3:** If any parameter in $\theta_{\text{cand}}$ contains `NaN` or `inf`:
     - Revert to $\theta_t$.
     - Emit event `REJECT_NONFINITE_PARAM`.
     - Skip update for batch $B_t$.

### 2.2 Mechanism 2: Bounded Parameter Updates ($\ell_\infty$ Box Projection)
To prevent unconstrained representation drift away from pre-trained source features, each updated BatchNorm affine parameter ($\gamma_l, \beta_l$) is projected into a hypercube centered at its source weight:

$$\gamma_l^{(t+1)} = \text{clamp}\left(\gamma_l^{\text{cand}}, \gamma_{l, \text{source}} - \epsilon_\gamma, \gamma_{l, \text{source}} + \epsilon_\gamma\right)$$
$$\beta_l^{(t+1)} = \text{clamp}\left(\beta_l^{\text{cand}}, \beta_{l, \text{source}} - \epsilon_\beta, \beta_{l, \text{source}} + \epsilon_\beta\right)$$

- **Candidate Hyperparameters (NOT Validated Constants):**
  - $\epsilon_\gamma \in [0.05, 0.50]$ (Initial probe candidate: $0.20$).
  - $\epsilon_\beta \in [0.05, 0.50]$ (Initial probe candidate: $0.20$).
- **Calibration Mandate:** These bounds must be calibrated on Hospital 1 (OOD development hospital) by auditing healthy adaptation trajectories. Hospital 2 test labels must never be used to tune $\epsilon$.

### 2.3 Mechanism 3: Immutable Source State & Fallback Engine
1. At initialization:
   ```python
   self.source_state = {
       k: v.detach().cpu().clone() 
       for k, v in model.state_dict().items()
   }
   ```
2. The `source_state` dictionary is marked read-only.
3. If an unrecoverable runtime exception or numerical error occurs during streaming adaptation:
   - Immediately reload `model.load_state_dict(self.source_state)`.
   - Freeze all model parameters (`requires_grad_(False)`).
   - Emit event `FALLBACK_TO_SOURCE`.
   - Complete remaining inference stream using the source model.

### 2.4 Mechanism 4: Event Telemetry Logger
Every adaptation step produces a structured record:
```json
{
  "batch_idx": 42,
  "status": "ACCEPTED", // or "REJECTED_NONFINITE", "CLAMPED", "FALLBACK"
  "coverage": 0.582,
  "loss_value": 0.0412,
  "grad_norm_before_clip": 0.381,
  "max_param_displacement_gamma": 0.045,
  "max_param_displacement_beta": 0.082,
  "num_clamped_tensors": 0
}
```

---

## 3. Explicit Non-Claims & Safety Boundaries

1. **No Non-Inferiority Guarantee:** MV-SafeAdapt does not guarantee that adapted test performance will match or exceed source performance. It guarantees only that:
   - Non-finite updates will never be committed to model weights.
   - BatchNorm affine parameters will never drift beyond distance $\epsilon$ from source weights.
2. **No Guaranteed Prevention of Single-Class Drift:** If labels on the test domain are heavily imbalanced, parameters can still shift within the $\epsilon$-ball toward lower sensitivity.
3. **Candidate Nature of Bounds:** The parameter bound $\epsilon = 0.20$ is a working hypothesis, not a mathematically derived constant.
