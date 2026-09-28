# Phase 11 — ResNet-50 backbone sensitivity

**Status: CODE READY, EXPERIMENTS NOT RUN.**  
B2 / B9 in the matrix. **Not in the 5-seed headline table.** One seed (42), same frozen TTA recipe.

## What exists

- `src/models/resnet50.py` — 2048-D GAP → Linear 512 → ReLU → Dropout 0.5 → 1 logit
- `configs/resnet50.yaml`
- `scripts/adapt.py --backbone resnet50`
- This GPU notebook: `notebooks/kaggle_resnet50_s42.py`

## Frozen protocol (do not change)

Same as ResNet-18: `bn_freeze_stats`, TTA lr=`1e-5`, N_MC=20, τ=70th percentile of **this** model’s hospital-1 U, EATA e=0.4, test hospital 2 once.

## Run (~5–8 h T4)

1. GPU T4, Internet on  
2. Import `kaggle_resnet50_s42.ipynb` (or run the `.py`)  
3. Train save-best on hospital 1; if OOM, set `BS_TRAIN=32`  
4. Then Tent / EATA / confidence / UTTA once  

## Pass rule

Source OOD val AUROC should be in a plausible range (typically ≥0.97). If Tent collapses, **keep the number**. Do not retune lr on hospital 2.

## Send back

`camelyon17_resnet50_full_s42.json` and the printed TEST block.

Compare to ResNet-18 seed 42 (source 0.935 / UTTA 0.951 / harm 0.003). If UTTA still ≈ confidence, that is the sensitivity conclusion.
