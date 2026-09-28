# Checklist vs this tree (2026-09-27)

`pytest tests/ -q` → **21 passed, 2 skipped** (WILDS PNG loader not installed; YAML leakage test **passes**).

There is **no ResNet-17**. Official: **ResNet-18**, 1-logit, dropout **0.5**, **Adam**.

## MUST FIX

| Item | Status |
|---|---|
| 1-logit vs 2-logit | **Fixed.** `ResNet18Classifier` raises if `num_classes != 1`. BEST.pt `classifier.3.weight` is `[1, 512]`. |
| dropout 0.5 vs 0.2 | **Fixed.** `OFFICIAL_DROPOUT_P = 0.5`. `source_resnet18.yaml` marked obsolete. |
| Adam vs AdamW | **Fixed.** `train_loop` and all TTA adapters use `torch.optim.Adam`. Tests assert no AdamW. |
| configs/code/checkpoints consistent | **Fixed.** Canonical: `configs/resnet18.yaml` + `authoritative_resnet18_utta.yaml` + `tau_by_seed.yaml`. |
| Seed 123 / 2024 / 7 / 99 checkpoints | **You add.** Download from Kaggle into `checkpoints/`. Cannot be reconstructed. Seed **42** is already here. |
| ResNet50UTTAMed import | **Fixed.** `from src.models import ResNet50UTTAMed`. Kaggle notebook imports it with a local fallback. |
| pytest | **Passes.** |
| scipy / pyarrow / fpdf2 | **In** `requirements.txt`, `environment.yml`, `reproducibility/requirements.txt`. |
| reproduction tau | **Fixed.** Not 0.05. `adapt.py` loads `configs/tau_by_seed.yaml` when `--tau` omitted. |
| reproducibility report checkpoint claim | **Fixed.** Report states only files that exist. |
| fake GitHub URL | **Removed.** No `github.com/anonymous/UTTA-Med`. |
| commit vs code | Local `git` on this tree. No public remote. |

## SHOULD FIX

| Item | Status |
|---|---|
| Raw five-seed predictions / U files | **Partial.** Seed-42 dumps if present under `results/dumps/`. Other seeds: keep Kaggle `dumps/s*.npz`. Too large for git. |
| Per-seed experiment manifests | **Added** `experiments/manifests/s{42,123,2024,7,99}.json`. |
| Per-seed run logs | Metrics JSON **is** the log. Dummy `experiments/.../metrics.json` with 0.0 must stay deleted. |
| Standalone WSI JSON seed 42 | **Present** `results/metrics/camelyon17_wsi_bootstrap_s42.json` and copy under `results/statistical/wsi_bootstrap/`. |
| Obsolete configs | `source_resnet18.yaml` marked obsolete (values aligned). |
| Legacy notebooks | `notebooks/LEGACY_NOTEBOOKS.md`. |
| Efficiency param counts | **40** BN affine tensors (ResNet-18), not 106. See `docs/EFFICIENCY_ANALYSIS.md`. |
| Paper numbers vs JSON | Table 1 matches `five_seed_summary` / locked JSONs. |

## You still copy (binary, not in git)

```
checkpoints/camelyon17_resnet18_source_s123_BEST.pt
checkpoints/camelyon17_resnet18_source_s2024_BEST.pt
checkpoints/camelyon17_resnet18_source_s7_BEST.pt
checkpoints/camelyon17_resnet18_source_s99_BEST.pt
```

Kaggle datasets: `camelyon17-resnet18-source-s{123,2024,7,99}-best-pt`.
