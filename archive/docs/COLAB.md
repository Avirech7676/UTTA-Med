# Colab run sheet — Gate B-FULL then Gate C (same T4)

## Before you start

1. Google account with ~12 GB free Drive space (checkpoint + HF cache optional).
2. Colab: **Runtime → Change runtime type → T4 GPU**.
3. Keep the browser tab open. Free Colab can kill idle / long sessions.

## Order (do not skip)

### 0. Mount Drive first
If the runtime dies after 2 hours, only Drive-saved checkpoints survive.

### 1. Gate B-FULL — `notebooks/01_colab_source_train.ipynb`
- Must print `cuda True` and a T4 (or better) name.
- First cell assert must not fail.
- HuggingFace download is ~10.7 GB **once**; later cells reuse cache.
- Train hospitals `{0,3,4}`, early-stop `{1}`, test `{2}` only after freeze.
- Wait until `camelyon17_resnet18_source_s42.json` exists.

**Do not close the runtime.**

### 2. Gate C — `notebooks/02_colab_tent_eata.ipynb` (same runtime)
- Loads the frozen source checkpoint.
- Continual Tent and EATA, BN-affine only, one pass, no shuffle.
- Writes `camelyon17_resnet18_tent_eata_s42.json`.

### 3. Download and send back
Upload only:

```
results/metrics/camelyon17_resnet18_source_s42.json
camelyon17_resnet18_tent_eata_s42.json   # if Gate C ran
```

No parquet. No `.venv`. Checkpoint `.pt` only if < 50 MB.

## Unzip gotcha

If `unzip` creates an extra folder, `cd` into the folder that contains `notebooks/`.

```
!unzip -q UTTA-Med-gateb-next.zip
!find . -name 01_colab_source_train.ipynb
```

## If Colab disconnects mid-train

Restart, mount Drive, confirm the latest `.pt` exists, and resume from that epoch
(or re-run — 2–4 hours is cheaper than a bad checkpoint).
Never continue TTA from `smoke_s42_max20.pt`.
