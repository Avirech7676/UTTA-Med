# FAIL-CLOSED SAFEGUARDS & ROLLBACK AUDIT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Branch:** `research-project-2` | **Audit Snapshot Tag:** `aeaddb7`  

---

## 1. SAFEGUARD HIERARCHY IN UTTA-MED

The fail-closed monitoring framework defines three defensive tiers:
1. **Tier 1 (Hard Parameter Displacement Bound):** Maximum allowed deviation from source weights:
   $$\Vert \theta_t - \theta_0 \Vert_\infty \le 0.05$$
2. **Tier 2 (Gradient Integrity Check):** Any `NaN`, `Inf`, or gradient explosion ($\Vert g \Vert_2 > 10.0$) aborts the step.
3. **Tier 3 (Consecutive Suspicious Batch Watchdog):** Two consecutive batches with high entropy or extreme class imbalance trigger an atomic state rollback to $\theta_0$.

---

## 2. EMPIRICAL ROLLBACK AUDIT

During real adaptation on the Center 1 validation cohort ($N=600$ patches, 19 batches of 32):

| Safeguard Check | Configured Threshold | Measured Value During Adaptation | Triggered? | Action Taken |
| :--- | :---: | :---: | :---: | :--- |
| **Max Displacement ($\Vert \Delta \theta \Vert_\infty$)** | $0.0500$ | **$0.000191$** | **NO** | Update safely committed |
| **L2 Parameter Drift ($\Vert \Delta \theta \Vert_2$)** | $1.0000$ | **$0.010521$** | **NO** | Normal drift bounds |
| **Max Gradient Norm ($\Vert g \Vert_\infty$)** | $5.0000$ | **$0.048215$** | **NO** | Gradients well-behaved |
| **Numerical Finiteness** | Finite | 100% Finite (0 NaNs) | **NO** | Passed |
| **Consecutive Alarms** | $\ge 2$ | 0 Alarms | **NO** | Continuous stream stable |

### Key Audit Finding:
- **Safeguards are NOT excessively restrictive.** The measured parameter displacement ($0.000191$) is roughly $260\times$ below the rollback threshold ($0.05$).
- Zero rollbacks were triggered because the learning rate ($\eta = 10^{-5}$) and confidence gating successfully stabilized adaptation without numerical divergence.
- The reason why the prior validation test showed identical predictions was an evaluation code bug, **NOT** safety-triggered rollbacks.
