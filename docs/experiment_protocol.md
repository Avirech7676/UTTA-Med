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
