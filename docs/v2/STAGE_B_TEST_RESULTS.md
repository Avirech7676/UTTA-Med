# Stage B: Automated Unit Test & Verification Results

**Document Version:** 2.0-TestReport  
**Date:** October 10, 2026  
**Auditor / Test Engineer:** Senior PyTorch Engineer & Independent ML Auditor  
**Test Suite:** `tests/test_safeadapt_v2.py` + Full Repository Regression Suite  
**Execution Environment:** Windows, Python 3.11.9, PyTorch 2.14.0+cpu, pytest-9.1.1  
**Hardware:** CPU Deterministic Execution  

---

## 1. Test Suite Summary

- **Total SafeAdapt Tests Run:** 14
- **Passed:** 14 (100%)
- **Failed:** 0
- **Regression Suite Run:** 35 items
- **Passed:** 33
- **Skipped:** 2 (Camelyon17 raw WSI filesystem verification, expected without local raw data)
- **Failed:** 0
- **Total Execution Time:** 29.59 seconds

---

## 2. SafeAdapt v2 Test Execution Details (`tests/test_safeadapt_v2.py`)

| Test Identifier | Tested Mechanism | Pass Criteria | Execution Result |
| :--- | :--- | :--- | :---: |
| `test_nan_loss_rejection` | Finite Loss Check | Step rejected; weights unmodified; finite fallback prediction returned | **PASSED** |
| `test_infinite_gradient_rejection` | Finite Gradient Check | Step rejected; weights unmodified; grad cleared; rejection logged | **PASSED** |
| `test_non_finite_post_step_parameter_rejection` | Post-Step Weight Check | Non-finite parameter step detected; weights rolled back to pre-step | **PASSED** |
| `test_adam_step_counter_and_optimizer_state_preservation` | Adam Moment & Counter Rollback | `step` counter and `exp_avg` / `exp_avg_sq` buffers strictly identical before & after rejected step | **PASSED** |
| `test_injected_invalid_optimizer_update_restores_exact_state` | Post-Step Injection & Exact State Restoration | Corrupted parameter after `step()`; verify `state_dict() == opt_state_before` and `model.state_dict() == pre_model_state` | **PASSED** |
| `test_source_state_immutability` | Source State Preservation | Adapt 5 steps; verify model adapted but `verify_source_immutability() == True` | **PASSED** |
| `test_bn_running_statistics_preservation` | Frozen BN Running Stats | Noisy batch adaptation; verify `running_mean` & `running_var` are bitwise identical | **PASSED** |
| `test_source_relative_parameter_projection_linf` | Bounded Box Projection ($\ell_\infty$) | Massive update step; verify all parameters satisfy $\|\theta - \theta_0\|_\infty \le \epsilon$ | **PASSED** |
| `test_exception_safe_rollback` | Exception Boundary | Injected `RuntimeError` during loss; verify clean rollback & source prediction returned | **PASSED** |
| `test_current_batch_source_fallback_on_rejection` | Safe Prediction Routing | Injected failure; verify predictions match frozen source model predictions | **PASSED** |
| `test_repeated_failures_trigger_fallback_mode` | Policy State Transition | Exceed `max_consecutive_rejections`; verify transition to fallback mode | **PASSED** |
| `test_reset_boundary_behavior` | Boundary Reset State | Call `reset_boundary(restore_source_weights=True)`; verify clean state | **PASSED** |
| `test_adapter_absence_of_target_label_inputs` | Zero Test Leakage Interface | Inspect `SafeAdaptTTA.adapt_batch(x)` signature; assert `y` absent | **PASSED** |
| `test_v1_artifacts_unmodified_checksum` | Repository Integrity | Verify checksum and content of `table_paired_statistics_final.csv` | **PASSED** |

---

## 3. Deep-Dive Forensic Test Verification

### 3.1 Verification of Post-`step()` Invalid Update Injection & Exact Restoration
- **Test:** `test_injected_invalid_optimizer_update_restores_exact_state`
- **Methodology:**
  1. Capture bitwise snapshots of model state dict and optimizer state dict:
     ```python
     weights_before = {k: v.clone() for k, v in model.state_dict().items()}
     opt_state_before = deepcopy(optimizer.state_dict())
     ```
  2. Monkeypatch `optimizer.step` to execute normally, then immediately overwrite adaptable parameter weights with `float('inf')`:
     ```python
     def corrupted_step():
         orig_step()
         with torch.no_grad():
             controller.adaptable_params[1].data.fill_(float('inf'))
     ```
  3. Execute `controller.step(x, compute_loss_fn=loss_fn)`.
  4. Assertions evaluated:
     ```python
     assert not accepted
     for k, v in model.state_dict().items():
         assert torch.equal(v, weights_before[k])
     assert optimizer.state_dict() == opt_state_before
     ```
- **Result:** **PASSED**. Both the PyTorch model weights and the Adam optimizer state dictionary (including internal step counters and momentum tensors) were restored bitwise to their exact pre-step states.

### 3.2 Verification of Adam Step-Counter Preservation
- **Test:** `test_adam_step_counter_and_optimizer_state_preservation`
- **Methodology:**
  1. Execute one valid step so that Adam instantiates internal momentum state dictionaries (`step = 1`).
  2. Record `optimizer.state[p]['step']`, `exp_avg`, and `exp_avg_sq`.
  3. Execute a subsequent step that fails with an injected `NaN` loss.
  4. Assertions evaluated:
     ```python
     assert optimizer.state[p]['step'] == step_before
     assert torch.equal(optimizer.state[p]['exp_avg'], exp_avg_before)
     assert torch.equal(optimizer.state[p]['exp_avg_sq'], exp_avg_sq_before)
     ```
- **Result:** **PASSED**. The Adam step counter remained frozen at `1` and was not incremented by the rejected candidate step.

### 3.3 Verification of Source State Immutability
- **Test:** `test_source_state_immutability`
- **Methodology:**
  1. Adapt the model over 5 sequential batches, verifying that model weights diverge from their initial values:
     $$\max_{p} \|\theta_{\text{adapted}} - \theta_{\text{source}}\| > 0$$
  2. Evaluate `controller.verify_source_immutability()`.
- **Result:** **PASSED**. The preserved source state remained strictly detached and byte-for-byte identical to the pre-adaptation snapshot.

---

## 4. Full Regression Test Run Output

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\avina\OneDrive\Desktop\UTTA-Med
configfile: pytest.ini
collected 35 items

tests/test_no_leakage.py::test_frozen_yaml_hospital_isolation PASSED     [  2%]
tests/test_no_leakage.py::test_wsi_no_overlap SKIPPED                    [  5%]
tests/test_no_leakage.py::test_hospital_separation SKIPPED               [  8%]
tests/test_no_leakage.py::test_protocol_constants_and_disjointness PASSED [ 11%]
tests/test_no_leakage.py::test_source_training_rejects_center_1 PASSED   [ 14%]
tests/test_no_leakage.py::test_source_training_rejects_center_2 PASSED   [ 17%]
tests/test_no_leakage.py::test_target_label_isolation_in_adaptation PASSED [ 20%]
tests/test_no_leakage.py::test_tau_and_threshold_isolation_to_center_1 PASSED [ 22%]
tests/test_no_leakage.py::test_canonical_transforms_unchanged PASSED     [ 25%]
tests/test_architecture.py::test_resnet18_one_logit_dropout_half PASSED  [ 28%]
tests/test_architecture.py::test_resnet18_rejects_two_class PASSED       [ 31%]
tests/test_architecture.py::test_resnet50_uttamed_import_and_shape PASSED [ 34%]
tests/test_architecture.py::test_source_optimizer_is_adam_not_adamw PASSED [ 37%]
tests/test_architecture.py::test_tta_optimizer_is_adam PASSED            [ 40%]
tests/test_architecture.py::test_resnet18_bn_affine_count PASSED         [ 42%]
tests/test_architecture.py::test_resnet50_bn_affine_count PASSED         [ 45%]
tests/test_architecture.py::test_official_tau_not_0p05 PASSED            [ 48%]
tests/test_architecture.py::test_canonical_camelyon17_transforms PASSED  [ 51%]
tests/test_architecture.py::test_benchmark_training_configuration PASSED [ 54%]
tests/test_architecture.py::test_canonical_configuration_and_reproducibility PASSED [ 57%]
tests/test_architecture.py::test_dependency_specifications_harmonized PASSED [ 60%]
tests/test_safeadapt_v2.py::test_nan_loss_rejection PASSED               [ 62%]
tests/test_safeadapt_v2.py::test_infinite_gradient_rejection PASSED      [ 65%]
tests/test_safeadapt_v2.py::test_non_finite_post_step_parameter_rejection PASSED [ 68%]
tests/test_safeadapt_v2.py::test_adam_step_counter_and_optimizer_state_preservation PASSED [ 71%]
tests/test_safeadapt_v2.py::test_injected_invalid_optimizer_update_restores_exact_state PASSED [ 74%]
tests/test_safeadapt_v2.py::test_source_state_immutability PASSED        [ 77%]
tests/test_safeadapt_v2.py::test_bn_running_statistics_preservation PASSED [ 80%]
tests/test_safeadapt_v2.py::test_source_relative_parameter_projection_linf PASSED [ 82%]
tests/test_safeadapt_v2.py::test_exception_safe_rollback PASSED          [ 85%]
tests/test_safeadapt_v2.py::test_current_batch_source_fallback_on_rejection PASSED [ 88%]
tests/test_safeadapt_v2.py::test_repeated_failures_trigger_fallback_mode PASSED [ 91%]
tests/test_safeadapt_v2.py::test_reset_boundary_behavior PASSED          [ 94%]
tests/test_safeadapt_v2.py::test_adapter_absence_of_target_label_inputs PASSED [ 97%]
tests/test_safeadapt_v2.py::test_v1_artifacts_unmodified_checksum PASSED [100%]

======================= 33 passed, 2 skipped in 29.59s ========================
```
