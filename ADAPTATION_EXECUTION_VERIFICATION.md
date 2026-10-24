# ADAPTATION EXECUTION & PIPELINE TRACE VERIFICATION REPORT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Branch:** `research-project-2` | **Dataset:** Camelyon17-WILDS Center 1 (N=600 validation patches)  
**Lead Investigator:** Senior Computational Pathology Researcher & PyTorch Test-Time Adaptation Engineer  

---

## 1. OFFICIAL ORDER OF OPERATIONS IN UTTA-MED / GATED-TTA

Inspection of [`src/tta/uncertainty_gated.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/uncertainty_gated.py#L22-L103) and [`src/tta/gates.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/gates.py#L15-L48) confirms the precise 10-step protocol:

1. **Model Loading:** The frozen ResNet-18 checkpoint (`checkpoints/camelyon17_resnet18_source_s*.pt`) is loaded into memory in evaluation mode.
2. **Batch Ingestion:** An unlabeled batch $x_b \in \mathbb{R}^{B \times 3 \times 96 \times 96}$ arrives sequentially from the clinical slide stream.
3. **Certainty Evaluation:** Confidence ($\max(p, 1-p)$) or MC-Dropout predictive variance ($\sigma^2_{\text{MC}}$ from 20 stochastic passes) is evaluated under `torch.no_grad()`.
4. **Admission Masking:** The admission condition is evaluated (e.g., $\text{conf} \ge 0.80$ or $\sigma^2_{\text{MC}} < \tau$). An indicator vector $w \in \{0, 1\}^B$ is generated.
5. **Loss Computation:** If $\sum w_i > 0$, `configure_tta_mode(model, bn_mode='bn_freeze_stats')` sets Batch Normalization layers to freeze running mean/variance while leaving affine parameters $(\gamma, \beta)$ trainable. The entropy loss $\mathcal{L}_{\text{ent}} = \frac{\sum w_i H(p_i)}{\sum w_i}$ is calculated.
6. **Backpropagation:** `loss.backward()` computes gradients specifically with respect to the 40 BN affine parameters ($2 \times 20$ layers).
7. **Optimizer Step:** `optimizer.step()` applies an Adam update with learning rate $\eta = 10^{-5}$.
8. **Safeguard Verification:** Stage B safeguards verify parameter displacement ($\Vert \Delta \theta \Vert_\infty \le 0.05$) and numerical finiteness.
9. **State Retention / Rollback:** If safeguards pass, the updated parameters are committed; otherwise, the state rolls back to the prior checkpoint.
10. **Post-Update Inference:** The adapted model produces the final prediction for batch $x_b$: $\hat{p} = \sigma(f_{\theta_{t}}(x_b))$. **Inference occurs AFTER the parameter update** on the current batch, exactly adhering to the standard online test-time adaptation protocol.

---

## 2. RESOLUTION OF THE IDENTICAL F1 PUZZLE ON CENTER 1

In the previous turn, Center 1 validation reported identical $F_1 = 0.8566$, identical false negatives (52), and zero harmful flips across all methods.

### Direct Root Cause:
In the prior evaluation script, the validation loop was evaluated with:
```python
adapted_p = np.where(mask, p_mean_val, p_frozen)
pred_after = (adapted_p >= 0.5).astype(int)
```
Where `p_mean_val` was the frozen Monte Carlo dropout mean on the linear head. Because the ensemble mean of linear head dropout has the identical sign as the frozen prediction (`p_mean_val >= 0.5` $\iff$ `p_frozen >= 0.5`), taking `np.where` did not change a single discrete prediction! **No actual gradient descent or optimizer steps were executed.**

### Corrected Real Execution:
When real online gradient adaptation is executed with `GatedTTA`:
- Batch Normalization affine parameters genuinely drift: $\Vert \Delta \theta \Vert_2 = 0.010521, \quad \Vert \Delta \theta \Vert_\infty = 0.000191$.
- **426 out of 600 patch probabilities change** ($|\Delta p| > 10^{-5}$).
- **2 hard predictions flip from incorrect to correct** (beneficial flips = 2, harmful flips = 0).
- Real adapted validation $F_1$ increases from **0.8566 to 0.8596**!

---

## 3. SUMMARY OF REAL ADAPTATION METRICS ACROSS POLICIES

| policy_name | f1_score | auroc | precision | recall | true_positives | false_negatives | false_positives | true_negatives | fn_change_vs_frozen | fp_change_vs_frozen | changed_probabilities_count | mean_abs_prob_diff | max_abs_prob_diff | hard_prediction_flips | harmful_flips | beneficial_flips | net_flip_benefit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_Frozen_ResNet18 | 0.8566 | 0.9601 | 0.8889 | 0.8267 | 248 | 52 | 31 | 269 | 0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| A1_Confidence_Gating_tau_0.80 | 0.8596 | 0.96 | 0.8953 | 0.8267 | 248 | 52 | 29 | 271 | 0 | -2 | 426 | 0.002239 | 0.035492 | 2 | 0 | 2 | 2 |
| A1_Confidence_Gating_tau_0.85 | 0.8596 | 0.96 | 0.8953 | 0.8267 | 248 | 52 | 29 | 271 | 0 | -2 | 436 | 0.001936 | 0.02825 | 2 | 0 | 2 | 2 |
| A2_Original_UTTA_Med_Percentile_70 | 0.8606 | 0.9607 | 0.8897 | 0.8333 | 250 | 50 | 31 | 269 | -2 | 0 | 491 | 0.002184 | 0.060323 | 2 | 0 | 2 | 2 |
| Candidate_CAD_Gate | 0.8596 | 0.96 | 0.8953 | 0.8267 | 248 | 52 | 29 | 271 | 0 | -2 | 438 | 0.00262 | 0.039796 | 2 | 0 | 2 | 2 |
| Matched_Budget_Confidence | 0.8581 | 0.9601 | 0.8921 | 0.8267 | 248 | 52 | 30 | 270 | 0 | -1 | 436 | 0.001622 | 0.028723 | 1 | 0 | 1 | 1 |

All results are fully documented in [`BN_PARAMETER_UPDATE_DIAGNOSTICS.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/BN_PARAMETER_UPDATE_DIAGNOSTICS.csv) and [`BEFORE_AFTER_ADAPTATION_COMPARISON.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/BEFORE_AFTER_ADAPTATION_COMPARISON.csv).
