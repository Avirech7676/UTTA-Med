# Stage B: Minimum Viable SafeAdapt Implementation Report

**Document Version:** 2.0-Implementation  
**Date:** October 10, 2026  
**Authors / Roles:** Senior PyTorch Engineer, Medical Imaging ML Researcher, Independent Scientific Auditor  
**Status:** COMPLETE (All CPU Acceptance Tests Passing)  
**Strict Mandate:** Zero modification of frozen v1 scientific artifacts, checkpoints, tables, or raw metrics. Zero GPU training or adaptation executed.

---

## 1. Executive Summary

Stage B of the SafeAdapt development plan has been successfully implemented under an isolated `src/v2/` namespace. The Minimum Viable SafeAdapt (MV-SafeAdapt) architecture introduces atomic update validation, non-finite rejection, source-relative $\ell_\infty$ parameter bounding, immutable source fallback, and structured telemetry.

All 14 deterministic CPU acceptance tests in `tests/test_safeadapt_v2.py` pass with **100% success (14/14)**, alongside the entire regression suite (33 passed, 2 skipped due to external WILDS dataset requirements).

---

## 2. Inventory of Created & Modified Files

### 2.1 Newly Created Implementation Files (v2 Namespace)
1. [`src/v2/__init__.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/__init__.py) — Root v2 research namespace.
2. [`src/v2/safeadapt/__init__.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/__init__.py) — SafeAdapt package interface.
3. [`src/v2/safeadapt/config.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/config.py) — Decoupled configuration dataclass with candidate parameters.
4. [`src/v2/safeadapt/telemetry.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/telemetry.py) — Privacy-preserving structured event logger.
5. [`src/v2/safeadapt/controller.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/controller.py) — Core atomic adaptation controller.
6. [`src/v2/safeadapt/adapter.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/adapter.py) — Label-free TTA interface wrapping PyTorch vision backbones.
7. [`configs/v2/safeadapt_minimal.yaml`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/configs/v2/safeadapt_minimal.yaml) — Minimal Stage B configuration template.
8. [`tests/test_safeadapt_v2.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/tests/test_safeadapt_v2.py) — Deterministic CPU unit test suite (14 test cases).
9. [`docs/v2/STAGE_B_IMPLEMENTATION_REPORT.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/docs/v2/STAGE_B_IMPLEMENTATION_REPORT.md) — This engineering and audit report.
10. [`docs/v2/STAGE_B_TEST_RESULTS.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/docs/v2/STAGE_B_TEST_RESULTS.md) — Detailed test execution report.

### 2.2 Status of Frozen v1 Scientific Artifacts
- **Checkpoints:** Unchanged. All five `.pt` source checkpoints remain byte-identical.
- **Results & Tables:** Unchanged. Table 1, Table 2, `table_paired_statistics_final.csv`, and all raw `results/metrics/full_s*.json` files were unedited.
- **Original Code:** Unchanged. `src/tta/tent.py`, `src/tta/eata.py`, `src/tta/uncertainty_gated.py`, and `src/tta/bn.py` were preserved without modification.

---

## 3. Architecture & Design Invariants

```
                            [ Test Batch x ]
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ 1. Immutable Source Snapshot & Fallback Check        │
        │    - If in fallback: evaluate with frozen source     │
        └──────────────────────────┬────────────────────────────┘
                                   │ Normal adaptation mode
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ 2. Pre-Step Snapshots (Model Weights & Adam State)   │
        └──────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ 3. Forward & Loss Computation                         │
        │    - Validate loss is finite (`isfinite(loss)`)       │
        │    - If NaN/Inf: ROLLBACK (Step skipped)              │
        └──────────────────────────┬────────────────────────────┘
                                   │ Finite loss
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ 4. Backward Propagation                               │
        │    - Validate gradients (`isfinite(p.grad)`)          │
        │    - If NaN/Inf: ROLLBACK (Step skipped)              │
        │    - Optional independent gradient clipping           │
        └──────────────────────────┬────────────────────────────┘
                                   │ Finite gradients
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ 5. Candidate Optimizer Step                           │
        │    - Validate candidate parameters & buffers          │
        │    - If NaN/Inf: ROLLBACK (Step skipped)              │
        │    - Optional L_inf projection: clamp to [θ_0 ± ε]    │
        └──────────────────────────┬────────────────────────────┘
                                   │ Finite parameters
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │ 6. Candidate Prediction Validation                    │
        │    - If predictions contain NaN: ROLLBACK             │
        │    - If valid: COMMIT UPDATE, reset counters          │
        └───────────────────────────────────────────────────────┘
```

### 3.1 Four Core Architectural Invariants

1. **Source State Immutability:**
   - At initialization, `SafeAdaptController` creates a detached CPU/device clone of the model's initial `state_dict`:
     ```python
     self._immutable_source_state = {
         k: v.detach().clone().cpu() for k, v in model.state_dict().items()
     }
     ```
   - This state is strictly read-only and serves as the ground-truth anchor for parameter displacement calculation, projection, and emergency fallback.
   - Verified by `verify_source_immutability()` in unit testing.

2. **Atomic Rollback & State Restoration:**
   - Prior to every candidate update, the controller snapshots:
     - `pre_model_state`: Cloned parameters of all adaptable modules.
     - `pre_optimizer_state`: Deep-copied state dictionary of the PyTorch optimizer (`deepcopy(optimizer.state_dict())`).
   - If an update is rejected at any stage (loss, gradient, parameter, prediction, or unhandled exception):
     - `model.load_state_dict(pre_model_state)` restores weights exactly.
     - `optimizer.load_state_dict(pre_optimizer_state)` restores Adam momentum buffers and ensures the step counter is **not incremented**.
   - Verified by injecting invalid parameters post-`step()` and testing exact bitwise restoration of model weights and Adam step counters.

3. **Bounded BatchNorm-Affine Updates:**
   - Adaptation scope is restricted to BatchNorm affine parameters ($\gamma, \beta$).
   - BatchNorm running statistics (`running_mean`, `running_var`) and Dropout layers are strictly frozen in `eval()` mode.
   - After optimizer execution, parameters are optionally projected back onto a source-relative hypercube:
     $$\theta_{t+1} = \text{clamp}(\theta_{\text{cand}}, \theta_{\text{source}} - \epsilon, \theta_{\text{source}} + \epsilon)$$
   - *Scientific Boundary:* The parameter bound $\epsilon = 0.20$ is configured as a candidate hyperparameter, decoupled from gradient clipping ($M_{\text{clip}} = 1.0$).

4. **Zero Test-Label Ingress:**
   - The adaptation interface `SafeAdaptTTA.adapt_batch(x)` accepts strictly the input tensor `x`.
   - Inspection of the call signature confirms no target labels `y` can be ingested during inference.

---

## 4. Verification Summary

| Invariant / Feature | Verification Method | Status |
| :--- | :--- | :---: |
| **NaN Loss Rejection** | Injected `loss = NaN`; assert weights unchanged & step rejected | **PASSED** |
| **Inf Gradient Rejection** | Hook injected `grad = Inf`; assert weights unchanged & step rejected | **PASSED** |
| **Non-Finite Param Rejection** | Monkeypatched step injecting `NaN`; assert weights restored | **PASSED** |
| **Adam Step Preservation** | Verify `optimizer.state[p]['step']` does not advance on rejection | **PASSED** |
| **Exact Optimizer Rollback** | Corrupt parameter post-step; assert `state_dict() == opt_state_before` | **PASSED** |
| **Source Immutability** | Adapt 5 steps; assert `verify_source_immutability() == True` | **PASSED** |
| **BN Running Stats Frozen** | Noisy batches; assert `running_mean` & `running_var` strictly identical | **PASSED** |
| **Parameter Bound Clamping** | Force massive gradient; assert `max(|θ - θ_0|) <= eps` | **PASSED** |
| **Exception-Safe Rollback** | Injected `RuntimeError` during loss; assert clean rollback & source fallback | **PASSED** |
| **Source Fallback Output** | Rejected step returns source predictions matching `forward_source(x)` | **PASSED** |
| **Repeated Failure Policy** | Trigger `max_consecutive_rejections`; verify transition to fallback mode | **PASSED** |
| **Boundary Reset** | Call `reset_boundary(restore_source_weights=True)`; verify clean state | **PASSED** |
| **Label-Free Interface** | Inspect `SafeAdaptTTA.adapt_batch` signature; assert `y` absent | **PASSED** |
| **v1 Artifact Checksum** | Verify integrity of `results/tables/table_paired_statistics_final.csv` | **PASSED** |

---

## 5. Remaining Limitations & Boundaries

1. **Finite Degradation Blind Spot:** Atomic validation catches non-finite values (`NaN`, `inf`). It does **not** prevent real-valued parameter drift into single-class negative attractors ($p \to 0$ with finite real numbers). That failure mode is addressed in Stage C (Collapse-Aware Controller).
2. **Candidate Bound Unvalidated:** The candidate bound $\epsilon = 0.20$ has not yet been calibrated against healthy adaptation trajectories on Hospital 1.
3. **No Clinical Non-Inferiority Claim:** Stage B provides numerical robustness and bounded parameter motion. It does not provide an unconditional guarantee that adapted performance will exceed or match source performance on unobserved clinical domains.

---

## 6. Formal Governance Recommendation

### **Recommendation: GO for Hospital-1 Development Validation**

- **Justification:** The core Stage B mechanics (atomic rollback, bounded projection, source preservation, and structured telemetry) are verified to be numerically robust, reproducible, and completely decoupled from frozen v1 artifacts.
- **Next Step:** Proceed to Stage C calibration on Hospital 1 (OOD development set) to tune candidate bounds ($\epsilon, M_{\text{clip}}$) and evaluate prediction-collapse monitors without touching Hospital 2 test data.
