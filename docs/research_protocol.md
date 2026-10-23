# Research Protocol Summary — UTTA-Med (v4 Frozen)

This document is the high-level pointer to the frozen design.

- **Research questions & hypotheses**: see `research_question.md`, `hypotheses.md`
- **Dataset & leakage rules**: see `dataset_protocol.md`, `leakage_rules.md`
- **Experiment design, matched coverage, seeds, statistics**: see `experiment_protocol.md`, `statistical_protocol.md`, `experiment_matrix.md`

## Non-negotiable Rules
1. Official WILDS Camelyon17 splits only. Hospital mapping verified from the loader and frozen in `configs/camelyon17.yaml`.
2. Target labels used **only** for final evaluation.
3. All TTA methods update only BatchNorm affine parameters.
4. Gate-control comparisons use matched coverage (preferred) and always report coverage.
5. Primary statistics are WSI-level bootstrap.
6. Minimum 5 seeds with fixed list.
7. Zero-accepted batches are explicitly skipped (no epsilon smoothing).
8. Negative or mixed results remain scientifically valid.

## Implementation Order
Follow the Phase-Gate Checklist in the Master Plan v4 (Stage 0 → Stage 23).
Start with infrastructure + PathMNIST pipeline, then Camelyon17 loader verification, leakage tests, source model, etc.
