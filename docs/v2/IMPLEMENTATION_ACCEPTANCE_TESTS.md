# SafeAdapt Implementation Acceptance Tests & Go/No-Go Governance

**Document Version:** 1.5-Governance  
**Date:** October 10, 2026  
**Auditor / Governance Authority:** Independent Machine Learning Auditor  
**Scope:** Pre-implementation acceptance criteria for Stage B (Minimum Viable SafeAdapt).  
**Execution Condition:** *Unit-testable on CPU/synthetic data prior to any GPU adaptation.*

---

## 1. Stage B Acceptance Test Suite Specification

The following automated test cases must be implemented in `tests/test_safeadapt_mvp.py` and pass with 100% success before any experimental adaptation is run:

### Test Case 1: Atomic Rejection of Non-Finite Loss
- **Condition:** In a mock forward pass, inject `loss = torch.tensor(float('nan'))`.
- **Expected Behavior:**
  - Optimizer step is NOT executed.
  - Zero backward gradients are propagated.
  - Model parameters remain identical to pre-step state (`torch.equal(p_before, p_after)`).
  - Event telemetry logs `REJECT_NONFINITE_LOSS`.
- **Pass Rule:** `assert event['status'] == 'REJECT_NONFINITE_LOSS'` and `assert max_diff == 0.0`.

### Test Case 2: Atomic Rejection of Non-Finite Gradients
- **Condition:** Simulate a backward pass that produces `p.grad[0] = float('inf')`.
- **Expected Behavior:**
  - `optimizer.step()` is skipped.
  - `optimizer.zero_grad(set_to_none=True)` clears the corrupted gradients.
  - Parameters remain unmodified.
  - Event telemetry logs `REJECT_NONFINITE_GRAD`.
- **Pass Rule:** `assert event['status'] == 'REJECT_NONFINITE_GRAD'`.

### Test Case 3: Bounded Parameter Projection ($\ell_\infty$ Box Constraint)
- **Condition:** Artificially apply an aggressive gradient update that pushes $\gamma$ from $1.0$ to $1.85$ when $\epsilon_\gamma = 0.20$.
- **Expected Behavior:**
  - Post-step clamp restricts $\gamma$ to $\min(1.85, 1.0 + 0.20) = 1.20$.
  - Event telemetry logs `CLAMPED` with displacement $\Delta = 0.20$.
- **Pass Rule:** `assert torch.all(param <= source_param + eps)` and `assert torch.all(param >= source_param - eps)`.

### Test Case 4: Immutable Source State Preservation
- **Condition:** Initialize SafeAdapt with pre-trained ResNet-18 weights. Run $100$ simulated noisy adaptation steps.
- **Expected Behavior:**
  - Model weights $\theta_t$ adapt dynamically.
  - The internal `source_state` dictionary remains byte-for-byte identical to initial checkpoint weights.
  - Invoking `fallback_to_source()` immediately restores the exact initial forward predictions.
- **Pass Rule:** `for k in source_state: assert torch.equal(source_state[k], initial_weights[k])`.

### Test Case 5: Zero Test-Label Access Assertion
- **Condition:** Audit the `SafeAdaptController` class interface.
- **Expected Behavior:**
  - The controller's `step()` and `adapt_batch()` methods do NOT accept target labels `y` in their call signatures.
  - All decisions (gating, loss calculation, clamping, fallback) depend strictly on `(x, model, source_state)`.
- **Pass Rule:** `assert 'y' not in inspect.signature(controller.step).parameters`.

---

## 2. Go / No-Go Decision Recommendation

### Current Status: **GO (For Stage B Minimal Viable Implementation)**

#### Rationale for "GO" on Stage B:
1. **Low Implementation Risk:** Stage B does not introduce complex heuristic triggers, running prevalence assumptions, or slide-level state machines. It introduces standard, mathematically well-defined safeguards: finite-value checks, bounded projection, and source state preservation.
2. **Defensible Scientific Grounding:** Reversible updates and bounded parameter manifolds are grounded in classical constrained optimization, avoiding premature claims of universal collapse prevention.
3. **Zero Test-Set Contamination:** All parameter bounds $\epsilon$ will be calibrated strictly on Hospital 1 validation trajectories. Hospital 2 remains completely isolated.

#### Explicit Conditions for Advancement to Stage C:
Advancement from Stage B to Stage C (Prediction-Collapse Monitoring) will require:
1. All 5 acceptance tests in `tests/test_safeadapt_mvp.py` pass.
2. Synthetic corruption tests on Hospital 1 demonstrate that Stage B safeguards eliminate all `NaN` occurrences without introducing training instability.
3. Confirmation that Stage B does not degrade healthy adaptation on Hospital 1 clean data.
