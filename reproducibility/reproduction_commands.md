# Official reproduction commands

There is no anonymous GitHub URL. Use the local folder that contains this file.

```bash
cd UTTA-Med
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

## Source training (one seed)

```bash
python scripts/train_baseline.py --data-root "$CAMELYON17_ROOT" --seed 42
```

Early-stop on hospital 1. Target hospital 2 is evaluated only after the checkpoint is frozen.

## Test-time adaptation (seed 42 example)

τ is **not** 0.05. Frozen seed-42 τ = `8.875e-06` (70th percentile of val U).
`adapt.py` loads `configs/tau_by_seed.yaml` when `--tau` is omitted for UTTA.

```bash
CKPT=checkpoints/camelyon17_resnet18_source_s42_BEST.pt
ROOT="$CAMELYON17_ROOT"

python scripts/adapt.py --checkpoint $CKPT --method tent --data-root $ROOT --lr 1e-5
python scripts/adapt.py --checkpoint $CKPT --method eata --data-root $ROOT --lr 1e-5 --e-margin 0.4
python scripts/adapt.py --checkpoint $CKPT --method utta --data-root $ROOT --seed 42
# equivalent explicit tau:
python scripts/adapt.py --checkpoint $CKPT --method utta --data-root $ROOT --tau 8.875383173290174e-06
```

| Seed | τ (70th pct val U) |
|---:|---|
| 42 | 8.875383173290174e-06 |
| 123 | 7.0986357059155125e-06 |
| 2024 | 1.2937300198245794e-05 |
| 7 | 1.1984909178863745e-05 |
| 99 | 1.8291444575879723e-05 |

## Data

HuggingFace `wltjr1007/Camelyon17-WILDS` parquet (not WILDS PNG `metadata.csv`).
