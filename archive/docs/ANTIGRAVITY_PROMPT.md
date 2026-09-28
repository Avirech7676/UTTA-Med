# UTTA-Med — Prompt for Antigravity (local agent)

Copy everything below the line into Antigravity. Work in the existing repo. Do not start a new project.

---

You are implementing **UTTA-Med** on this Windows machine. Master Plan v4 is frozen. Do not change the research design.

## Goal

Uncertainty-aware test-time adaptation for binary tumor/non-tumor patch classification under cross-hospital shift on Camelyon17-WILDS.

Core method: MC-Dropout predictive variance gate. Adapt only if `U(x) < τ`. If a batch has zero accepted samples, skip the optimizer step (no epsilon trick).

## Machine paths (do not invent others)

```
Project:  C:\Users\avina\OneDrive\Desktop\UTTA-Med
Data:     C:\Users\avina\OneDrive\Desktop\Camelyon17-Data
```

Data layout is **HuggingFace parquet**, not WILDS PNG:

```
Camelyon17-Data\
  README.md
  data\
    train-00000-of-00014.parquet ... train-00013-of-00014.parquet
    validation-00000-of-00003.parquet ...
    test-00000-of-00004.parquet ...
```

This is `wltjr1007/Camelyon17-WILDS`. Columns: `image, label, center, image_id, patient, node, x_coord, y_coord, slide`.

Hospital IDs must be **verified from the files**, then written to `configs/camelyon17.yaml`. Official WILDS convention (confirm, do not hard-code blindly):

- train parquet → centers {0, 3, 4} = SOURCE
- validation parquet is MIXED (~68k): center **1** = OOD val (τ / HPs only); centers {0,3,4} = id_val
- test parquet → center {2} = TARGET

Never use target (test) labels for training, adaptation, τ, or model selection.

## Immediate environment fixes

The project `.venv` is broken (`python -m pip` → No module named pip). Prefer one of:

1. `python -m ensurepip --upgrade` inside the venv, then `python -m pip install -r requirements.txt`, OR
2. Deactivate the venv and use the user Python 3.14 that already has torch/pandas, OR
3. Recreate venv with Python 3.11 or 3.12 if 3.14 blocks wheels (`pyarrow`, `grad-cam`).

Core packages: `torch torchvision numpy pandas pyarrow scikit-learn matplotlib pyyaml tqdm pillow pytest`. Skip `pytorch-grad-cam` / `monai` until later. PyPI name for Grad-CAM is `grad-cam`.

**Fix `src/data/__init__.py`:** make it import-light (docstring only). An old `__init__.py` that imports `Camelyon17ParquetDataset` currently crashes inspect.

## Absolute scientific rules (never break)

1. Official split logic only. No custom re-split of patches.
2. Target labels: final evaluation only.
3. τ and all HPs: OOD validation hospital only (center 1).
4. All TTA methods update **BatchNorm affine (γ, β) only**.
5. Mandatory methods: B1 source-only, B4 Tent, B5 EATA, B6 random-gate, B7 confidence-gate, B10 entropy-gate, B8 UTTA-Med (MC-Dropout variance).
6. Matched coverage for gated methods; report coverage on every gated row.
7. Empty gate → skip batch (no backward, no step, log it).
8. Primary statistics: WSI/slide-level bootstrap (`bootstrap_ci_by_slide`), not patch-level.
9. Minimum 5 seeds: `{42, 123, 2024, 7, 99}`.
10. Do not add ViT, CLIP, LLM, segmentation, CT/MRI.

## What the repo already contains

`src/models/resnet18.py` (head: Linear-ReLU-Dropout-Linear, MC-Dropout).
`src/tta/` Tent, EATA, gates, uncertainty_gated.
`src/evaluation/` classification, adaptation, statistics.
`scripts/train_baseline.py`, `adapt.py`, `evaluate.py`, `inspect_local_data.py`.
`src/data/camelyon17.py` + `parquet_camelyon17.py` (parquet backend). If those files are missing or broken, implement them.

Public API:

```python
dataset, root, data_dir = load_camelyon17(data_root=r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data")
train = get_split(dataset, "train", train=True)
val   = get_split(dataset, "val", train=False)    # center == 1
test  = get_split(dataset, "test", train=False)
idval = get_split(dataset, "id_val", train=False)
```

`__getitem__` must return `(image, y, metadata)` like WILDS.

## Work order (stop at each gate and summarize)

### Gate A — Data
- Fix env + `__init__.py`.
- `python scripts/inspect_local_data.py --data-root "C:\Users\avina\OneDrive\Desktop\Camelyon17-Data"`
- Freeze `configs/camelyon17.yaml` with real `domain_ids`, counts, `leakage_free: true`.
- Confirm train ∩ val ∩ test WSI IDs are empty.
- Expected ballpark: train ~302k, val(center1) ~35k, id_val ~34k, test ~85k.

### Gate B — Source-only ResNet-18
- Train on train split, early-stop on OOD val (center 1). Never peek at test during training.
- Save `checkpoints/camelyon17_resnet18_source_s42.pt`.
- Then evaluate test once. Log AUROC, AUPRC, F1, sens, spec, ECE, Brier.
- Compute shift: id_val vs test.

### Gate C — Tent + EATA
- Same BN-affine scope, same target stream, same seeds.
- Continual adaptation on unlabeled test stream; labels only after adaptation for metrics.

### Gate D — MC-Dropout + UTTA-Med
- Validate U↑ ⇒ Error↑ on val before gating.
- τ sweep on val only; freeze τ; then test.
- Zero-accepted-batch skip.
- Gate controls at matched coverage: random / confidence / entropy vs MC-Dropout.

### Gate E — Reliability + stats
- Harmful adaptation table, coverage for all gates, reliability diagrams.
- `bootstrap_ci_by_slide()`.
- 5-seed matrix when compute allows.

## Coding standards

- Small, testable modules. Match existing repo layout.
- Windows: `num_workers` 0–2 for DataLoader.
- Every run writes `experiments/<id>/{config.yaml,metrics.json,git_commit.txt}` and `results/metrics/<id>.json`.
- Do not fabricate metrics. Empty tables until real runs finish.

## When a human reviews

After each gate, print: what changed, exact commands, key metrics, files created, any protocol deviation. Zip the repo excluding `.venv`, checkpoints larger than needed, and parquet data. Keep `configs/camelyon17.yaml` and `results/`.

Start now at Gate A.
