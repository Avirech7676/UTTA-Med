# Phase 1 — Zero-GPU uniqueness upgrade

Run **on your machine** inside the UTTA-Med project root (where `results/data/wsi_bootstrap/` lives).

## Requirements

```text
results/data/wsi_bootstrap/s{seed}_{method}.npz
  seeds:   42, 123, 2024, 7, 99
  methods: source, tent, eata, random, confidence, entropy, utta
  keys:    y, p, p_before, slide
```

Python 3.10+ with `numpy` only (matplotlib optional for figures).

## Run

```powershell
cd <project-root>

# copy these scripts into the project
mkdir scripts\upgrade -Force
copy <path-to>\scripts_upgrade\*.py scripts\upgrade\

python scripts\upgrade\phase1_all.py
```

Or run modules separately:

```powershell
python scripts\upgrade\slide_level_from_npz.py
python scripts\upgrade\coverage_risk_curves.py
python scripts\upgrade\tent_collapse_case_study.py
```

## Outputs

| File | Content |
|---|---|
| `results/statistical/phase1_slide_level.json` | Per-slide AUROC/F1 + slide-macro mean±sd |
| `results/statistical/phase1_coverage_risk.json` | Risk / AUROC vs keep-fraction (reject option) |
| `results/statistical/phase1_tent_collapse.json` | Seeds 7/99 + per-slide ΔAUROC vs source |
| `results/tables/slide_level_5seed.csv` | Flat table for Excel / paper |
| `figures/phase1/*.png` | Optional plots if matplotlib is installed |

## Rules

- Does **not** load checkpoints or Camelyon parquet.
- Does **not** retune τ or re-run TTA.
- Recomputes metrics from official dumps only.
- Keep seeds 7 and 99 in every table.
