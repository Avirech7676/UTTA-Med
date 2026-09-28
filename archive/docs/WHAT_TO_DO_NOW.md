# What to do now

Mandatory 5-seed science is locked. Remaining optional methods are coded.
They need **one Kaggle T4 run** on the existing seed-42 checkpoint. No retrain.

## 1. Kaggle (this zip)

1. New notebook, GPU T4, Internet on.
2. Add Input: seed-42 BEST.pt dataset.
3. Upload `UTTA-Med-remaining-optional.zip`, unzip.
4. Open `kaggle_remaining_optional.ipynb` → Run all (~45–70 min).
5. Download `/kaggle/working/camelyon17_remaining_optional_s42.json`.

If SAR/EATA-C/DLTTA collapse, **keep the number**. Do not retune.

## 2. After the JSON

Return it here. Then:

- Optional-baseline rows (seed 42 only, labeled as such)
- Temperature ECE/Brier table
- Figure 9 (corruptions)
- Manuscript LAST

## 3. Do not

- Retrain seeds 7/99
- Add seed 10
- Drop Tent collapses
- Tune on hospital 2
- Start Camelyon16 / MIDOG++ / ViT / ResNet-50
