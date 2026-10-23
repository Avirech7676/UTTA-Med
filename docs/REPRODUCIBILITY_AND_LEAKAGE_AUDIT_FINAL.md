# UTTA-Med Stage 13 — Reproducibility & Leakage Close-Out

Date: 2026-09-27
Protocol: Camelyon17-WILDS, ResNet-18 primary study, five seeds {42,123,2024,7,99}.

## 1. Final protocol invariants

- Source training: centers 0, 3, 4.
- OOD validation / hyperparameter selection: center 1 only.
- Target test: center 2 only.
- Target labels are evaluation-only and are not passed to adaptation.
- TTA learning rate: 1e-5.
- MC-Dropout passes: 20 for the primary experiment.
- tau: per-seed 70th percentile of validation predictive variance.
- Adaptation scope: BatchNorm affine parameters only.
- BatchNorm running statistics: frozen (`bn_freeze_stats`).
- Source checkpoint is reset/anchored before each comparable method.
- Target stream: 85,054 patches, 1,329 batches at batch size 64.

## 2. Leakage checks

| Check | Status | Evidence |
|---|---|---|
| Target labels excluded from adaptation | PASS | Adaptation receives x only; y is retained for post-adaptation reporting. |
| tau selected only on OOD validation | PASS | Per-seed tau is the 70th percentile of validation U. |
| Target set used only for final evaluation | PASS | Five-seed close-out states Hospital 2 only; no retuning on seeds 7/99. |
| Official hospital split retained | PASS | Source 0/3/4, validation 1, target 2. |
| Same BN-affine update scope | PASS | Tent/EATA/UTTA use BN-affine-only scope in the frozen protocol. |
| BN running statistics frozen | PASS | Current final TTA artifacts explicitly record `bn_mode=bn_freeze_stats`. |
| No target-based threshold retuning | PASS | `no_retune_on_7_99=true`; target coverage is reported after frozen thresholds. |

## 3. Reproducibility artifacts

- Five source checkpoints: seeds 42, 123, 2024, 7, 99.
- Five-seed master statistics JSON.
- Per-seed delta table.
- WSI bootstrap / per-slide bootstrap artifacts.
- Reliability figures.
- MC-pass sensitivity figure/data.
- Adaptation-step sensitivity figure/data.
- Grad-CAM audit artifacts.

## 4. Important audit correction

An earlier pipeline audit contained historical configuration details in which BN layers were described as training with batch statistics and source-training code paths differed from the frozen research configuration. Those historical details are **not** the final protocol. The current primary protocol is explicitly `bn_freeze_stats`, TTA LR=1e-5, MC=20, and per-seed validation-only tau selection. The final paper must cite only the frozen protocol.

## 5. Remaining repository-level evidence to attach before public release

- [ ] Record the exact final Git commit hash.
- [ ] Record final `requirements.txt` / environment lock hash.
- [ ] Record exact reproduction commands.
- [ ] Run a clean-environment smoke reproduction from the final commit.
- [ ] Verify no credentials/tokens or machine-specific secrets are in the release archive.

## Stage 13 disposition

**Scientific leakage/protocol audit: PASS.**
**Public-release reproducibility audit: CONDITIONAL until commit/environment/clean-reproduction metadata are attached.**
