# Kaggle — exact clicks (replace Colab)

Kaggle gives ~30 GPU hours/week (T4 or P100). That is enough for Gate C, and for a source retrain if the `.pt` was lost.

## A. One-time setup (5 minutes)

1. Open [https://www.kaggle.com](https://www.kaggle.com) and sign in (Google login is fine).
2. Phone verification is often required before GPU. Do it under **Settings → Phone**.
3. Top left: **Create** → **New notebook**.
4. Right sidebar:
   - **Accelerator** = `GPU T4` or `GPU P100`
   - **Internet** = **On**
   - **Persistence** = Files only (optional)

## B. Optional but better: upload the source checkpoint

Only if you still have `camelyon17_resnet18_source_s42.pt` (~46 MB).

1. Kaggle home → **Datasets** → **New Dataset**
2. Upload the `.pt` file
3. Title e.g. `utta-med-source-s42`
4. Create
5. In the notebook: **Add Input** → your dataset

If you do **not** have the `.pt`, skip this. The notebook retrains source-only on Kaggle GPU (~1–2 hours), then runs Tent + EATA.

## C. Load the notebook

**File → Import notebook** → upload `kaggle_utta_med_gate_bc.ipynb`

or **File → Import notebook** from GitHub if you host it.

Then **Run All**.

## D. What it does

| Step | Time on T4 | Output |
|---|---|---|
| Download HuggingFace Camelyon17 | 10–20 min first run | cached |
| Source train (only if no `.pt`) | 1–2 h | `camelyon17_resnet18_source_s42.pt` |
| Source eval | ~10 min | `..._source_s42.json` |
| Tent + EATA | ~20–40 min | `..._tent_eata_s42.json` |

Keep the tab open. Kaggle sessions last several hours.

## E. Download results

Right sidebar → **Output** → download:

- `camelyon17_resnet18_source_s42.json`
- `camelyon17_resnet18_tent_eata_s42.json`
- `camelyon17_resnet18_source_s42.pt` (save this locally and to Drive)

Paste both JSON files back in chat.

## F. Do not

- Turn Internet off (HuggingFace download will fail)
- Use a CPU notebook
- Close the tab during training
- Tune Tent/EATA on the test split
