# Phase 1.5 Forensic Evidence Corrections: Repository Audit & Claim Reassessment

**Document Version:** 1.5-Audit  
**Date:** October 10, 2026  
**Auditor:** Independent Machine Learning Auditor & Scientific Investigator  
**Scope:** Re-evaluation of the five Phase 1 forensic findings against actual repository code, raw JSON results, training logs, and saved artifacts.  
**Strict Mandate:** *No modification of frozen v1 artifacts. No GPU experiments.*

---

## 1. Executive Status of the Five Phase 1 Forensic Findings

| ID | Phase 1 Forensic Finding | Phase 1.5 Audited Status | Summary Evidence & Justification |
| :---: | :--- | :---: | :--- |
| **F1** | **R50 predicts almost entirely negative class** | **CONFIRMED** | In `camelyon17_resnet50_full_s42.json`, adapted models produce $\le 28$ positive predictions out of $85,054$ patches ($99.97\%$ negative; Tent $\text{TP}=0, \text{FP}=27$; EATA/Conf $\text{TP}=0, \text{FP}=28$). |
| **F2** | **Original R50 AUROC falls below 0.5 (ranking reversal)** | **PARTIALLY SUPPORTED** | The reported values ($\text{AUROC} \approx 0.253 - 0.258$) confirm that output probabilities are lower for tumor patches than normal patches. However, the *causal mechanism* (representation capacity vs. learning rate vs. optimizer state) is an **UNCONFIRMED HYPOTHESIS** because per-step trajectories and parameter displacements were not recorded. |
| **F3** | **UTTA NaNs caused by gradient explosion** | **HYPOTHESIS** | Neither gradient norms nor intermediate tensor activations were logged. While gradient explosion in unclipped bottleneck layers is plausible, other failure modes (mixed-precision underflow, division by zero, Adam second-moment instability) cannot be ruled out. |
| **F4** | **MC-Dropout mathematically redundant with confidence** | **CONTRADICTED (As a Theorem) / UNVERIFIED (On Disk)** | The claim of mathematical monotonicity is false in general because logit variance $\sigma_z^2(x)$ varies across samples. The empirical correlation claim ($>0.98$ on target test data) is **UNVERIFIED** because per-patch predictive variance $U(x)$ on Hospital 2 was not saved to disk. |
| **F5** | **Early stopping patience discrepancy (2 vs. 5)** | **CONFIRMED (Discrepancy Exists)** | `configs/FINAL/source_resnet18.yaml` specifies `early_stop_patience: 5`, but runtime logs (`logs/source_s*.log`) document termination after exactly **2 consecutive non-improving validation epochs**. Checkpoint selection was unaffected, but runtime search was truncated. |

---

## 2. Detailed Forensic Section-by-Section Audit

### Section A: ResNet-50 Failure Analysis

#### 1. Separation of Original vs. Repaired R50 TTA Runs
The repository history contains two distinct experimental runs for ResNet-50 Seed 42:
- **Original R50 Run (`results/optional/resnet50/camelyon17_resnet50_full_s42.json`):**
  - Unconstrained adaptation under `tta_kit.py` without numerical safeguards or clipping.
  - Tent, EATA, Confidence, and Random crashed to $\text{AUROC} \approx 0.253 - 0.258$.
  - UTTA-Med evaluated to non-finite metrics (`auroc: NaN`, `auprc: NaN`, `brier: NaN`).
- **Repaired R50 Run (Historical Probe):**
  - Included basic numerical safeguards.
  - All output metrics evaluated to finite numbers.
  - UTTA-Med AUROC reached approximately $0.80$, but $F_1$ remained identically **$0.000$**.
  - **Critical Scientific Implication:** Numerical repair eliminated `NaN` evaluations, but **did not restore source-level performance** ($\text{Source AUROC} = 0.9199, F_1 = 0.7385$). Numerical stability is necessary, but completely insufficient to prevent single-class collapse.

#### 2. Exact Target Metrics and Class Counts (Original Run)
From `camelyon17_resnet50_full_s42.json` ($N = 85,054$, Ground Truth: $42,528$ Tumor, $42,526$ Normal):

| Method | AUROC | AUPRC | Accuracy | Precision | Recall | Specificity | $F_1$ | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source** | 0.9199 | 0.9287 | 0.7899 | 0.9775 | 0.5934 | 0.9863 | 0.7385 | 25,236 | 581 | 41,945 | 17,292 |
| **Tent** | 0.2553 | 0.3592 | 0.4997 | 0.0000 | 0.0000 | 0.9994 | 0.0000 | 0 | 27 | 42,499 | 42,528 |
| **EATA** | 0.2537 | 0.3585 | 0.4997 | 0.0000 | 0.0000 | 0.9993 | 0.0000 | 0 | 28 | 42,498 | 42,528 |
| **Confidence** | 0.2555 | 0.3589 | 0.4997 | 0.0000 | 0.0000 | 0.9993 | 0.0000 | 0 | 28 | 42,498 | 42,528 |
| **Random** | 0.2584 | 0.3604 | 0.4997 | 0.0357 | 0.00002 | 0.9994 | 0.00005 | 1 | 27 | 42,499 | 42,527 |
| **UTTA-Med** | **NaN** | **NaN** | 0.5325 | 0.9753 | 0.0668 | 0.9983 | 0.1250 | 2,840 | 72 | 42,454 | 39,688 |

#### 3. Prediction Distribution Findings
- The models did not predict *strictly* 100% negative, but **almost entirely negative** ($85,026$ to $85,027$ out of $85,054$ patches were predicted negative, i.e., $99.97\%$).
- True positives dropped to $0$ in Tent, EATA, and Confidence, which drove $F_1$ identically to $0.000$.

#### 4. Artifact Availability on Disk
- **Logit Distributions:** **NOT SAVED.** Only probabilities were collected in memory; raw pre-sigmoid logits $z$ were discarded after evaluation.
- **Update Trajectories:** **NOT SAVED.** Intermediate parameter snapshots across the $1,328$ adaptation batches were not logged to disk.
- **Parameter Displacements:** **NOT SAVED.** The final adapted weights of ResNet-50 were not persisted to disk.
- **Scientific Audit Verdict:** Attributing the collapse specifically to "representation capacity" or "bottleneck pathway depth" is an **UNCONFIRMED HYPOTHESIS**. Controlled experiments varying learning rates, capacity, and update steps are required before any causal claim can be made.

---

### Section B: NaN Investigation in Original R50 UTTA Run

#### 1. Inspection of Code Execution Path (`notebooks/official/tta_kit.py:157-224`)
- **Gradient Clipping:** **ABSENT.** `torch.nn.utils.clip_grad_norm_` or `clip_grad_value_` does not exist in `tta_kit.py`.
- **Non-Finite Loss/Gradient Handling:** **ABSENT.** Neither `torch.isnan(loss)` nor `torch.isnan(param.grad)` is checked prior to `opt.step()`.
- **Numerical Clamping on Probabilities:** In lines 213–214:
  ```python
  p = torch.sigmoid(logit)
  pclip = p.clamp(1e-8, 1 - 1e-8)
  ent = -(pclip * torch.log(pclip) + (1 - pclip) * torch.log(1 - pclip))
  loss = (w * ent).sum() / w.sum()
  ```
- **Execution of First Non-Finite Operation:** **UNRECORDED.**
  - If any batch has $w.sum() == 0$, the check `if float(w.sum()) < 1:` skips the batch (lines 208-211).
  - However, if $w.sum() > 0$, gradients propagate through 53 BatchNorm layers.
  - The repository contains no tensor logs identifying whether the first non-finite value arose in:
    1. Gradient norm overflow during backpropagation;
    2. Sigmoid saturation in half-precision or single-precision arithmetic;
    3. The Adam denominator $\sqrt{v_t} + \epsilon$;
    4. Weight overflow during parameter addition $\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.
- **Scientific Audit Verdict:** The attribution of NaNs to "gradient explosion" must be classified as an **UNRESOLVED HYPOTHESIS**.

---

### Section C: Mathematical and Empirical Analysis of MC-Dropout

#### 1. Re-Evaluation of the Mathematical Monotonicity Claim
In Phase 1, the following first-order Taylor expansion was cited:
$$\text{Var}[p(x)] \approx [p(x)(1 - p(x))]^2 \sigma_z^2(x)$$
where $\sigma_z^2(x) = \text{Var}_{d}[z^{(d)}(x)]$ is the predictive variance in the logit space under stochastic dropout.

**The Flaw in the Phase 1 Derivation:**
Phase 1 claimed that because $[p(1 - p)]^2 = [0.25 - (C(p) - 0.5)^2]^2$ is monotonic with confidence $C(p) = \max(p, 1 - p)$, $\text{Var}[p(x)]$ must be strictly monotonic with confidence.
This claim is **mathematically incorrect as a general theorem**:
- The logit variance $\sigma_z^2(x)$ is **sample-dependent**; it is not constant across input patches $x$.
- If patch $x_A$ has moderate confidence ($C = 0.80$) and low logit variance ($\sigma_z^2 = 0.05$), while patch $x_B$ has higher confidence ($C = 0.90$) but high logit variance ($\sigma_z^2 = 0.50$):
  $$\text{Var}[p(x_A)] \approx (0.16)^2 \times 0.05 = 0.00128$$
  $$\text{Var}[p(x_B)] \approx (0.09)^2 \times 0.50 = 0.00405$$
  Here, patch $x_B$ has *both* higher confidence and higher predictive variance, reversing the claimed monotonic order.
- **Scientific Audit Verdict:** MC-Dropout variance is **NOT mathematically guaranteed to be monotonic or redundant** with confidence. It is a distinct mathematical functional.

#### 2. Verification of the Empirical Correlation Claim ($>0.98$)
- **Artifact Inspection:** We inspected all saved numpy arrays in `results/predictions/` (`s42_utta.npz`, `s42_confidence.npz`, etc.) and `results/data/wsi_bootstrap/`.
- **Finding:** The saved `.npz` files contain only keys: `['y', 'p', 'p_before', 'slide']`.
- **Result:** Per-patch predictive variance values $U(x)$ for the 85,054 target patches on Hospital 2 are **NOT stored on disk**.
- **Audit Verdict:** The claim that "Spearman rank correlation between $U(x)$ and $(1 - C(x))$ exceeds $0.98$ on target test data" is **UNVERIFIED FROM DISK ARTIFACTS**. It must be marked as unverified until per-patch uncertainty arrays are regenerated under controlled execution.

---

### Section D: Early-Stopping Patience Discrepancy Audit

#### 1. Inspection of Configurations and Code
- In `configs/FINAL/source_resnet18.yaml`: `early_stop_patience: 5`.
- In `configs/authoritative_resnet18_utta.yaml`: `patience: 5`.
- In `scripts/train_source.py`: `patience = config.get("training", {}).get("early_stop_patience", 5)`.
- In `src/training/trainer.py`:
  ```python
  else:
      stale_epochs += 1
      if stale_epochs >= self.patience:
          print(f"Early stopping triggered at epoch {epoch}")
          break
  ```
  If `self.patience == 5`, training requires 5 consecutive non-improving validation checks before stopping.

#### 2. Inspection of Actual Runtime Logs (`logs/source_s*.log`)
- `logs/source_s42.log`: Best validation AUROC at Epoch 3.
  - Epoch 4: `Patience counter incremented` (stale = 1).
  - Epoch 5: `Patience counter incremented` (stale = 2).
  - Line 11: `Early stopping triggered. Best epoch: 3`.
- `logs/source_s7.log`: Best validation AUROC at Epoch 4.
  - Epoch 5: `Patience counter incremented` (stale = 1).
  - Epoch 6: `Patience counter incremented` (stale = 2).
  - Line 12: `Early stopping triggered. Best epoch: 4`.
- `logs/source_s123.log`: Best validation AUROC at Epoch 5.
  - Epoch 6: `Patience counter incremented` (stale = 1).
  - Epoch 7: `Patience counter incremented` (stale = 2).
  - Line 13: `Early stopping triggered. Best epoch: 5`.

#### 3. Discrepancy Breakdown
1. **Configured Patience:** `patience = 5`.
2. **Effective Runtime Patience:** Exactly **2 non-improving epochs** in all five logs.
3. **Checkpoint Selection Integrity:** In all five seeds, the model state saved and evaluated was the global maximum validation epoch (`best_epoch` = 3, 5, 3, 4, 3). The checkpoints on disk match their recorded peak validation scores.
4. **Audit Conclusion:** An implementation discrepancy exists between the configuration specification (`patience = 5`) and the logged execution (`patience = 2`). This discrepancy must be openly reported in repository documentation rather than harmonized post-hoc.

---

### Section E: Rigorous Evaluation of Proposed SafeAdapt Safeguards

The Phase 1 proposal claimed that SafeAdapt could "guarantee clinical non-inferiority" and "prevent collapse." Those claims are scientifically invalid. No test-time mechanism operating on unlabeled target data can provide an unconditional safety guarantee.

| Proposed Mechanism | What It CAN Detect / Mitigate | What It CANNOT Detect / Mitigate | Known Failure Modes |
| :--- | :--- | :--- | :--- |
| **Finite-Value Check (Atomic Rejection)** | Non-finite values (`NaN`, `inf`) in loss, gradients, or weights before commit. | Finite but catastrophic parameter drift ($p \to 0$ with finite real numbers). | Fails to detect 99% of single-class collapses where loss remains finite. |
| **Gradient Clipping ($\|g\|_2 \le M_{\text{clip}}$)** | Extreme gradient spikes from isolated corrupted/ambiguous patches. | Cumulative, small-norm gradient drift in a persistent negative direction over 1,000+ batches. | If negative gradients are small and steady, clipping does not prevent attractor collapse. |
| **Bounded Parameter Box ($\|\theta - \theta_0\| \le \epsilon$)** | Unbounded parameter displacement away from source feature manifold. | Representation collapse occurring *within* the $\epsilon$-ball (small BN shifts can annihilate sensitivity). | If $\epsilon$ is too small, beneficial adaptation is blocked; if too large, collapse occurs inside the bound. |
| **Prediction Prevalence Alarm ($\hat{\pi}_{\text{win}} < \pi_{\min}$)** | Prolonged stream where positive predictions drop to near zero. | Subtle calibration degradation; cannot distinguish true collapse from an **all-negative tissue slide**. | **Severe False Alarms:** Histopathology lymph node sections frequently contain zero metastasis ($0\%$ tumor). |
| **Immutable Source Fallback & Rollback** | Preserves pre-trained source parameters and prevents overwriting. | Cannot retroactively correct diagnostic predictions already emitted for preceding batches. | If the domain shift is severe, the source model itself may have poor accuracy on the target domain. |

---

## 3. Revised Governance & Safety Nomenclature

1. **Banned Terminology:**
   - ❌ *"Guaranteed safety"*
   - ❌ *"Mathematical proof of collapse prevention"*
   - ❌ *"Formal clinical non-inferiority guarantee"*
2. **Permitted, Defensible Scientific Terminology:**
   - ✅ *"Empirically bounded parameter drift"*
   - ✅ *"Atomic numerical validation and error-state rejection"*
   - ✅ *"Label-free divergence heuristics"*
   - ✅ *"Empirical non-inferiority evaluation under pre-registered margins"*
