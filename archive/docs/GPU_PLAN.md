# UTTA-Med — GPU plan (Gate B is blocked without this)

Local machine: Intel Iris Xe, `torch.cuda.is_available() == False`.
A 20-batch / 1-epoch smoke is **not** a source-only baseline.

Do **not** run Tent / EATA / UTTA-Med on `camelyon17_resnet18_source_s42.pt`
from the smoke run. Those numbers cannot go in the paper.

## What the smoke actually proved

| Check | Result |
|---|---|
| Parquet loader + hospital split | Pass |
| Train loop, BCE, checkpoint write | Pass |
| Target labels used only at eval | Pass |
| Cross-hospital drop exists even after 20 batches | Pass (directionally) |
| Publication source model | **Fail — undertrained** |

Expected full source-only (ImageNet-pretrained ResNet-18, Camelyon17):
ID-val AUROC typically ~0.95–0.99, target AUROC often ~0.85–0.95
depending on training length. Smoke numbers (ID 0.965 / target 0.911)
are **not** to be quoted until a full run reproduces them.

## Hardware options (pick one)

1. **Google Colab (recommended, free T4)**
   - Runtime → Change runtime type → T4 GPU
   - ~2–4 hours for 20 epochs + eval with AMP
   - Notebook: `notebooks/01_colab_source_train.ipynb`
2. **Kaggle Notebooks** — similar free GPU quota, same notebook
3. **University / cloud NVIDIA GPU** — run `scripts/train_baseline.py` as-is
4. **Do not** train 20 epochs on CPU (~30 hours, not worth it)

## Colab data options

A. HuggingFace streaming/cache (easiest):
   `datasets.load_dataset("wltjr1007/Camelyon17-WILDS")`
   First run downloads ~10.7 GB into Colab disk (Colab Pro / high-RAM helps).

B. Upload your local parquet folder to Google Drive and mount it.
   Path should contain `data/train-*.parquet`.

## After a real source checkpoint exists

Only then:

```
Gate C  Tent (BN-affine only) on target stream
Gate C2 EATA (BN-affine only, entropy filter)
Gate D  MC-Dropout validation on OOD val, then UTTA-Med + gate controls
```

Seeds for headline runs stay `{42, 123, 2024, 7, 99}`.
First full source model: seed 42 only.
