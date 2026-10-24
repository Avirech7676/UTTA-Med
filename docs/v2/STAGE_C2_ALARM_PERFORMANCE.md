# Stage C2: Collapse Alarm Performance & Stress Testing

**Status:** Completed & Validated  
**Date:** October 2026  
**Auditor:** Independent Scientific Auditor & PyTorch Reliability Engineer  
**Dataset:** Camelyon17-WILDS (Hospital 1 & Negative Control)  
**Study Scope:** Exploratory / Provisional ($N_{\mathrm{WSI}} = 10$)  

---

## 1. Executive Summary

This report evaluates the performance, false-alarm rate, latency, and stress-testing dynamics of the Stage C2 calibrated SafeAdapt controller. 

The collapse monitor uses a multi-signal decision policy incorporating:
1. Logit variance collapse ($\sigma_z^2 < 0.08$)
2. Extreme continuous and binary source-adapted disagreement ($\Delta_{\mathrm{bin}} > 0.35$)
3. Predictive entropy collapse ($\bar{H} < 0.05$ with extreme positive fraction)
4. Escalating persistence thresholds ($k_{\mathrm{rollback}} = 2$, $k_{\mathrm{fallback}} = 4$)

---

## 2. False Alarm Rate on Genuinely Negative Slides

A major clinical risk in digital pathology test-time adaptation is that an uncalibrated monitor could misinterpret a legitimately tumor-free lymph node slide (0% tumor patches) as an "all-negative collapse", generating false alarms and blocking valid adaptation.

### Empirical Validation on Negative Specimens:

| Specimen Identifier | Center / Hospital | Patch Count ($N$) | Ground Truth Tumor Prevalence | Updates Attempted | Updates Rejected | False Alarm Rate | Alarm State |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Slide 6** (Negative Control) | Center 0 | 64 | 0.00% | 2 | 0 | **0.0%** | `NORMAL` |
| **Slide 15** (Hospital 1 Assessment) | Center 1 | 64 | 0.00% | 2 | 0 | **0.0%** | `NORMAL` |
| **Slide 14** (Hospital 1 Calibration) | Center 1 | 64 | 3.12% | 2 | 0 | **0.0%** | `NORMAL` |

**Conclusion:** Across all negative and low-prevalence whole-slide images, the false-alarm rate was **0.0%**. Because the guardrail requires source-adapted disagreement or logit variance collapse before flagging an alarm, healthy predictions on normal tissue remain unpenalized.

---

## 3. Alarm Latency & Escalation Dynamics

The multi-stage escalation policy operates deterministically across sliding evaluation windows:

```
Window t: Suspicious signal detected
  ├── consec = 1: AlarmLevel.WARNING (Telemetry logged, update accepted)
  ├── consec = 2: AlarmLevel.REJECT_UPDATE (Atomic pre-step rollback executed, source output returned)
  ├── consec = 3: AlarmLevel.REJECT_UPDATE (Atomic pre-step rollback executed, source output returned)
  └── consec = 4: AlarmLevel.FALLBACK (Slide permanently locks to frozen source, adaptation terminated)
```

### Measured Operational Characteristics:
- **Detection Latency:** Exactly 1 window to issue a warning; 2 windows ($\le 64$ patches) to trigger active rollback.
- **Rollback Overhead:** Pre-step state restoration requires $< 0.1\text{ ms}$ on CPU, as model weights and optimizer buffers are restored in-memory without disk I/O.
- **Inference Guarantee on Rejection:** When an update is rejected, the current batch is re-evaluated through the frozen source model, guaranteeing that the returned predictions are never contaminated by the corrupted gradient step.

---

## 4. Robustness & Failure Mode Stress Testing

### 4.1 Synthetic Single-Class Collapse Injection
To verify active-mode intervention on the real ResNet-18 pathology model, a severe adversarial target shift was injected during test-time adaptation:

$$\mathcal{L}_{\mathrm{collapse}} = \sum_{i} (\hat{z}_i - 50.0)^2$$

This loss forces all logits toward $+50.0$ ($\hat{p} \to 1.0$, logit variance $\to 0$).

#### Step-by-Step Controller Response:
- **Step 0:** $\sigma_z^2 = 0.00057$, $\Delta_{\mathrm{bin}} = 0.0$, `consecutive = 1` $\to$ **`AlarmLevel.WARNING`** (Update accepted, warning logged).
- **Step 1:** $\sigma_z^2 = 0.00053$, `consecutive = 2` $\to$ **`AlarmLevel.REJECT_UPDATE`** (Update **rejected**, atomic rollback executed).
- **Step 2:** $\sigma_z^2 = 0.00054$, `consecutive = 3` $\to$ **`AlarmLevel.REJECT_UPDATE`** (Update **rejected**, atomic rollback executed).
- **Step 3:** `consecutive = 4` $\to$ **`AlarmLevel.FALLBACK`** (Update **rejected**, permanent source fallback triggered).
- **Step 4:** In fallback mode $\to$ Update bypassed, frozen source evaluated directly.
- **Step 5:** In fallback mode $\to$ Update bypassed, frozen source evaluated directly.

**Stress Test Result:** 5 out of 6 candidate steps were rejected, and the controller successfully transitioned to permanent source fallback.

### 4.2 Batch-Size Sensitivity
The controller was stress-tested across varying test batch sizes $B \in \{16, 32, 64\}$:
- **Batch Size 16:** 4/4 batches accepted, 0 alarms, numerically stable.
- **Batch Size 32:** 2/2 batches accepted, 0 alarms, numerically stable.
- **Batch Size 64:** 1/1 batch accepted, 0 alarms, numerically stable.

The monitor dynamically scales its rolling window and sample sufficiency checks, remaining stable across small and large batch regimes.

---

## 5. Critical Scientific Governance Caveats

> [!CAUTION]
> **Scientific Notice Regarding Synthetic Stress Testing:**
> Successful detection and containment of synthetic single-class collapse in controlled CPU experiments does **not** prove guaranteed prevention of real-world clinical failure. Real-world distribution shifts can manifest as subtle, slide-level calibration drift that does not trip coarse variance or disagreement thresholds.
>
> The Stage C2 controller is an empirical safety guardrail designed to catch catastrophic collapse; it does not claim clinical non-inferiority or replace rigorous prospective validation.
