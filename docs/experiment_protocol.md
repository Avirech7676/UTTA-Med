# Experiment Protocol — UTTA-Med

**Frozen experimental design. Changes require explicit justification and version bump.**

## Mandatory Baselines
| ID  | Method                          | Notes |
|-----|---------------------------------|-------|
| B1  | ResNet-18 source-only           | Lower bound |
| B4  | ResNet-18 + Tent                | Standard TTA |
| B5  | ResNet-18 + EATA                | Strong selective TTA (frozen second baseline) |
| B6  | Random-gated TTA                | Gate control |
| B7  | Confidence-gated TTA            | Gate control |
| B10 | Entropy-gated TTA               | Gate control |
| B8  | ResNet-18 + UTTA-Med (MC-Dropout) | Proposed method |

## Update-Scope Parity (Critical)
All TTA methods (Tent, EATA, Random/Confidence/Entropy gates, UTTA-Med) update **only BatchNorm affine parameters** (γ, β).  
If the original EATA paper updates more parameters, restrict it to BN-affine for fair comparison and document the deviation.

## Matched-Coverage Protocol (Preferred)
1. Select τ for UTTA-Med on OOD validation only.
2. Record the resulting coverage on the target stream.
3. Tune Random / Confidence / Entropy gate thresholds on OOD validation so each achieves approximately the same coverage.
4. Compare performance at matched coverage.
5. Always report actual coverage for every gated method.

## Multi-Seed Protocol
- Minimum 5 seeds for all headline comparisons.
- Fixed seed list: `{42, 123, 2024, 7, 99}` (extend if 10 seeds are run).
- Same seeds used for every method.

## Statistical Protocol
- Primary CIs and significance tests: **WSI-level bootstrap** (resample slides, then take all patches of those slides).
- Implement as `src/evaluation/statistics.py :: bootstrap_ci_by_slide()`.
- Patch-level bootstrap is secondary sensitivity analysis only.

## τ Selection
- Sweep on OOD validation only.
- Freeze τ before any target-test evaluation.
- Report full τ–coverage–performance–calibration–harm curves.

## Zero-Accepted-Batch Handling
```python
if w.sum() == 0:
    log("batch skipped: zero samples passed gate")
    continue
loss = (w * entropy_per_sample).sum() / w.sum()
```
No epsilon smoothing.

## Artifact Convention
Every run produces:
```
experiment/
├── config.yaml
├── command.txt
├── git_commit.txt
├── metrics.json
├── predictions.parquet
├── uncertainty.parquet
├── checkpoint.pt
├── confusion_matrix.png
├── reliability.png
└── logs/
```

## Official Image Resolution and Hyperparameter Reconciliation

1. **Official Input Resolution ($96\times96$):**
   - The authoritative Camelyon17-WILDS patch resolution is strictly $96\times96$ RGB with ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - Labels are determined by whether the central $32\times32$ tissue area contains metastatic carcinoma.

2. **Exploratory $224\times224$ Preprocessing:**
   - Any references to $224\times224$ preprocessing belong strictly to preliminary drafts, standard torchvision ImageNet defaults, or exploratory foundation model investigations (e.g., NCT-CRC-HE-100K 9-class tiles or ResNet-50).
   - $224\times224$ preprocessing was **never** used in the official five-seed Camelyon17 benchmark runs.

3. **Learning Rate Provenance:**
   - **Official Source Pretraining:** Adam with $\text{lr} = 10^{-4}$ and $\text{weight\_decay} = 10^{-4}$ (max 20 epochs, early stopping patience 5 on Hospital 1 AUROC).
   - **Official TTA:** Single-pass ($k=1$), Adam with $\text{lr} = 10^{-5}$ under `bn_freeze_stats` (updating only BatchNorm $\gamma, \beta$).
   - Preliminary learning rates such as Tent's original paper default ($\text{lr} = 10^{-3}$) caused immediate collapse on histopathology and must not be confused with the frozen official $\text{lr} = 10^{-5}$.

4. **Archival Integrity:**
   - All legacy and preliminary configuration iterations remain safely quarantined under `archive/configs/` for transparent auditability.

5. **Best Checkpoint Epoch vs. Final Training Epoch:**
   - **Best Checkpoint Epochs (Saved & Evaluated Weights):** The optimal checkpoints selected by peak Hospital 1 validation AUROC are **Epoch 3** (Seeds 42, 2024, 99), **Epoch 4** (Seed 7), and **Epoch 5** (Seed 123). These weights are stored in `checkpoints/*_BEST.pt` and recorded in `results/metrics/full_s*.json` (`best_epoch`).
   - **Final Training Termination Epochs:** After identifying the best checkpoint, the training loop continued for subsequent validation checks until early stopping triggered termination at **Epoch 5** (Seeds 42, 2024, 99), **Epoch 6** (Seed 7), and **Epoch 7** (Seed 123), as documented in `logs/source_s*.log`.
   - Therefore, while the *best checkpoints* span epochs 3–5, the *training loops* physically terminated at epochs 5–7.
