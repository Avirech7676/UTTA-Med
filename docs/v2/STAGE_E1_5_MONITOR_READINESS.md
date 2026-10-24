# Stage E1.5: Collapse-Monitor Activation and Detector Readiness Audit

**Auditor:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Audit Complete — Critical Detector Discrepancy Identified & Reconciled  
**Target:** SafeAdapt Collapse Monitor Architecture & Telemetry Integration

---

## 1. Executive Summary

During the pre-Stage E2 audit of Stage E1 results, an independent scrutiny of detector mechanics revealed a critical configuration discrepancy:
1. **The Frozen E1 Collapse Monitor Never Evaluated an Alarm:** Due to an omitted configuration parameter (`min_samples_for_alarm`), the monitor's minimum sample threshold defaulted to 100 patches, while `window_size` was capped at 10 patches. Because `w = min(n_samples, 10)` could never satisfy $w \ge 100$, `is_sufficient_samples` was permanently `False` on all batches across all slides.
2. **Telemetry Re-assessment:** The "0 alarms" reported in Stage E1 Arm D was not evidence of a low false-alarm rate on Hospital 1; it was the direct consequence of the detector being effectively muted by contradictory window/sample thresholds.
3. **Parameter Bound Re-assessment:** The observed parameter displacement ($L_\infty = 2.003 \times 10^{-5}$ vs. bound $0.05$, $0.040\%$ utilization) confirms that adaptation was exceptionally conservative on Hospital 1, but does *not* prove that the bound actively averted an impending parameter explosion.
4. **Continuous-Stream Re-assessment:** The tumor-ascending continuous stream demonstrated order-induced calibration degradation ($+8.1\%$ ECE) in an offline stress-test scenario. This confirms that per-WSI resets eliminate cross-slide carryover in this specific experiment, but does not constitute a generalized clinical safety proof across all pathology settings.

---

## 2. Collapse-Monitor Mechanics and Stage E1 Activation Analysis

### A. Denominator of Rolling Window $W$
Code inspection of `src/v2/safeadapt/monitor.py` (`SlideState.append_batch` and `_compute_window_stats`) confirms:
- $W$ counts **individual patch predictions**, not batches or whole slides.
- When a batch of 32 patches is processed, 32 distinct probability scalars, source probability scalars, and logit values are pushed into the slide buffer deque:
  $$\text{buffer\_length} = \sum_{b=1}^B N_{\text{patches}}^{(b)}$$
- The sliding evaluation window slices the most recent $w = \min(\text{buffer\_length}, W)$ patches.

### B. The Contradictory Window/Threshold Defect in Stage E1
In Stage C2 calibration (`docs/v2/STAGE_C2_CALIBRATION_REPORT.md`), the monitor was defined with:
- `window_size`: 50 patches
- `min_samples_for_alarm`: 20 patches

However, in `configs/v2/safeadapt_e1_frozen.yaml`, the specification was:
```yaml
monitor:
  mode: ACTIVE
  window_size: 10
  min_logit_variance: 0.08
  max_disagreement_rate: 0.35
  min_predictive_entropy: 0.05
  consecutive_suspicious_for_rollback: 2
  consecutive_suspicious_for_fallback: 4
```
Notice that `min_samples_for_alarm` was **omitted**.
In `CollapseMonitorConfig` (`src/v2/safeadapt/monitor.py`), the class definition specifies:
```python
@dataclass
class CollapseMonitorConfig:
    window_size: int = 500
    min_samples_for_alarm: int = 100
```
Consequently:
1. `config.window_size` = 10
2. `config.min_samples_for_alarm` = 100 (unoverridden class default)
3. In `_compute_window_stats`:
   $$w = \min(N_{\text{available}}, 10) \le 10$$
   $$\text{is\_sufficient\_samples} = (w \ge 100) \equiv \mathbf{False} \quad \forall \, N_{\text{available}}$$
4. In `update()`:
   ```python
   if not stats.is_sufficient_samples:
       stats.alarm_level = AlarmLevel.NORMAL
       return UpdateResult(AlarmLevel.NORMAL, stats, "insufficient_window_samples")
   ```

### C. Monitoring Decision Count per WSI in Stage E1
Across all 5 slides in Stage E1:
- **Total batches evaluated:** 10 (2 per WSI)
- **Total alarm evaluations attempted:** 10
- **Total valid alarm evaluations performed:** **0**
- **Reason returned on all 10 batches:** `"insufficient_window_samples"`

**Conclusion:** The Stage E1 report must transparently disclose that Arm D operated identically to Arm C not because the active monitor actively monitored and cleared the batches, but because sample insufficiency bypassed alarm evaluation entirely.

---

## 3. Impact of WSI Resets on Monitor Accumulation

Even if `min_samples_for_alarm` had been set to 20 or 50, per-WSI boundary resets profoundly limit detector opportunities on short patch streams:
- In the Stage E1 exploratory denominator (64 patches / WSI, 2 batches of 32):
  - **Batch 1 (t = 1):** 32 patches seen. If $\min \ge 50$, Batch 1 cannot be evaluated.
  - **Batch 2 (t = 2):** 64 patches seen. If $\min = 20$ or $50$, Batch 2 is evaluated.
  - **WSI Boundary Reset:** Buffer wiped to 0.
- Under a 2-batch-per-slide protocol, the monitor has at most **1 evaluation opportunity per slide**.
- Because `consecutive_suspicious_for_rollback = 2` requires 2 consecutive alarm windows, an atomic rollback could **never** trigger within a 2-batch slide under independent resets.
- For a realistic whole-slide image containing 1,000 to 5,000 patches (30 to 150 batches), $W=50$ and $\min=20$ provide dozens of continuous sliding evaluation windows, allowing multi-window persistence tracking to operate as designed.

---

## 4. Deterministic Integration Verification of Monitor Escalation

To verify that the underlying SafeAdapt detector logic functions as designed when thresholds are mathematically attainable, deterministic CPU tests were conducted on `SafeAdaptController` and `CollapseMonitor`:

### Test Configuration:
- `window_size` = 50 patches
- `min_samples_for_alarm` = 20 patches
- `consecutive_suspicious_for_rollback` = 2
- `consecutive_suspicious_for_fallback` = 4
- Batch size = 32 patches

### Observed Escalation Dynamics:
1. **Normal Batch ($t=1$, 32 patches):** $w=32 \ge 20$. $\text{is\_sufficient} = \text{True}$. Logit variance and entropy normal.  
   $\to$ `AlarmLevel.NORMAL` (`consecutive_suspicious = 0`). Update accepted.
2. **Induced Single-Class Collapse Batch ($t=2$, 64 patches):** Weights driven to constant extreme negative logits ($\sigma(z) \approx 0$).  
   $\to$ Severe disagreement with source + logit variance collapse.  
   $\to$ `AlarmLevel.WARNING` (`consecutive_suspicious = 1`). Single isolated window: update provisionally allowed.
3. **Sustained Collapse Batch ($t=3$, 64 patches):** Collapse persists into second consecutive window.  
   $\to$ Escalation threshold reached ($c \ge 2$).  
   $\to$ `AlarmLevel.REJECT_UPDATE` (`consecutive_suspicious = 2`). **Update rejected, atomic rollback to pre-step state executed.**
4. **Subsequent Collapse Batches ($t=4, 5$):**  
   $\to$ At $c \ge 4$, controller transitions to `AlarmLevel.FALLBACK`. **Irreversible fallback to immutable source state engaged.**

This confirms that the detector, rejection, rollback, and fallback mechanisms are functionally sound in PyTorch, but require non-contradictory configuration settings.

---

## 5. Re-Assessment of Stage E1 Claims

### A. Parameter Displacement Bound
- **Reported:** $L_\infty = 2.003 \times 10^{-5}$ vs. bound $0.05$ ($0.040\%$ utilization).
- **Corrected Scientific Interpretation:** The parameter bound did not "contain" an active explosion. Rather, standard Adam with $\text{lr}=10^{-5}$ over 2 steps was naturally inert on benign Hospital-1 data. The bound serves as an unbreached outer safety envelope, not an active clamp.

### B. Continuous-Stream Stress Test
- **Reported:** ECE increased from 0.0786 to 0.0850 (+8.1%) and F1 dropped from 0.8632 to 0.8596 under tumor-ascending order.
- **Corrected Scientific Interpretation:** Ordering slides by ground-truth tumor burden is an artificial, adversarial offline stress test. It proves that continuous, unreset adaptation exhibits path dependency and cumulative drift under extreme sequence skew. It does *not* prove that failure will occur under normal clinical orderings. The scientifically defensible conclusion is:
  > *"Per-WSI boundary resets eliminate sequence-dependent parameter accumulation and calibration drift in offline stress testing, providing a reproducible guarantee of slide-order invariance."*

---

## 6. Configuration Versioning and Protocol Rectification

To maintain strict scientific provenance without modifying frozen v1 artifacts:
- **`configs/v2/safeadapt_e1_frozen.yaml` remains unedited** as the historical artifact of Stage E1.
- A new, reconciled configuration **`configs/v2/safeadapt_e2_frozen.yaml`** is defined for Hospital 2 target validation, explicitly restoring the calibrated Stage C2 monitor parameters:
  ```yaml
  monitor:
    mode: ACTIVE
    window_size: 50
    min_samples_for_alarm: 20
    min_window_samples: 20
    min_logit_variance: 0.08
    max_disagreement_rate: 0.35
    min_predictive_entropy: 0.05
    consecutive_suspicious_for_rollback: 2
    consecutive_suspicious_for_fallback: 4
  ```
