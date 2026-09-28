# Phase 20 — Repository Cleanup Report

Date: 2026-09-28

## Completed cleanup

- Removed stale `scripts/verify_seed42.py` with obsolete result paths.
- Removed the old `train_baseline.py` entry point from the active scripts path; `train_source.py` is canonical.
- Archived superseded/duplicate configuration files under `archive/configs/`.
- Archived development-stage and historical documentation under `archive/docs/`.
- Archived all historical Kaggle/Colab notebooks under `archive/notebooks/`.
- Archived the duplicate top-level figure collection under `archive/figures/`.
- Archived optional/historical ResNet-50, BCD and collapsed-Tent artifacts under `archive/optional/` and `archive/historical/`.
- Archived one-off/phase-specific scripts under `archive/scripts/`.
- Removed Python bytecode and pytest cache artifacts.
- Updated all five experiment manifests to `checkpoint_present: true`.
- Updated the local reproduction guide to the final Hugging Face Parquet protocol.
- Updated the statistical protocol to distinguish five-seed inference from WSI-level bootstrap inference.
- Updated README, setup and reproducibility instructions for the cleaned release.
- Added `docs/FINAL_RELEASE_STATUS.md` and this report.

## Active canonical paths

```text
configs/FINAL/
src/
scripts/
tests/
results/
reproducibility/
paper/
```

## Checkpoint validation

All five checkpoints load successfully and contain a 1-logit classifier with shape `[1, 512]`:

- seed 42 — epoch 3 — validation AUROC 0.9792571865
- seed 123 — epoch 5 — validation AUROC 0.9823153897
- seed 2024 — epoch 3 — validation AUROC 0.9853831424
- seed 7 — epoch 4 — validation AUROC 0.9778726056
- seed 99 — epoch 3 — validation AUROC 0.9773151747

SHA256 values are recorded in `reproducibility/ARTIFACT_MANIFEST.csv`.

## Verification performed in this build environment

- `tests/test_metrics.py`, `tests/test_model.py`, `tests/test_tta.py`, `tests/test_architecture.py`, `tests/test_no_leakage.py`: **15 passed, 2 skipped**.
- `scripts/verify_config.py`: **PASS**.
- `scripts/verify_step2_guardrails.py`: **PASS — all 5 guardrails**.
- `scripts/verify_consistency.py`: **PASS — all 5 seeds consistent**.
- `scripts/audit_5seeds.py`: **PASS — all 7 methods across all 5 seeds**.
- Full `pytest -q` could not complete in this build container because `pyarrow` is not installed there; the locked reproduction environment explicitly includes `pyarrow==16.1.0`.
- `scripts/verify_protocol.py` likewise requires `pyarrow` and was not runnable in this container.

The dependency limitation above is an environment limitation of this packaging run, not a change to the frozen experiment protocol.
