# No Colab GPU — what to do (Gate C still runs)

Colab T4 quota is exhausted. **Do not retrain the source model.**
Gate C (Tent / EATA) is one pass, BN-affine only. CPU is enough.

## 0. Rescue the checkpoint (do this first)

The `.pt` lives in the Colab VM until that runtime is deleted.

1. Colab → **Runtime → Change runtime type → CPU** (not GPU).
2. If `/content/checkpoints/camelyon17_resnet18_source_s42.pt` still exists, download it:
   ```python
   from google.colab import files
   files.download('/content/checkpoints/camelyon17_resnet18_source_s42.pt')
   files.download('/content/results/metrics/camelyon17_resnet18_source_s42.json')
   ```
3. Also copy to Drive if Drive mounts:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p /content/drive/MyDrive/UTTA-Med/runs
   !cp /content/checkpoints/camelyon17_resnet18_source_s42.pt /content/drive/MyDrive/UTTA-Med/runs/
   !cp /content/results/metrics/camelyon17_resnet18_source_s42.json /content/drive/MyDrive/UTTA-Med/runs/
   ```

Put the `.pt` here on the PC:

`C:\Users\avina\OneDrive\Desktop\UTTA-Med\checkpoints\camelyon17_resnet18_source_s42.pt`

If the file is gone, you must retrain on **Kaggle** (section 3). Do not train 20 epochs on CPU.

## 1. Gate C on your PC (CPU) — recommended tonight

From the UTTA-Med folder, venv on:

```powershell
cd C:\Users\avina\OneDrive\Desktop\UTTA-Med
.\.venv\Scripts\Activate.ps1

python scripts\adapt.py `
  --checkpoint checkpoints\camelyon17_resnet18_source_s42.pt `
  --method tent `
  --data-root "C:\Users\avina\OneDrive\Desktop\Camelyon17-Data" `
  --split test --seed 42 --batch-size 32 --num-workers 0 --lr 1e-3 --steps 1

python scripts\adapt.py `
  --checkpoint checkpoints\camelyon17_resnet18_source_s42.pt `
  --method eata `
  --data-root "C:\Users\avina\OneDrive\Desktop\Camelyon17-Data" `
  --split test --seed 42 --batch-size 32 --num-workers 0 --lr 1e-3 --steps 1 --e-margin 0.4
```

Expect ~30–90 minutes **per** method on Iris Xe. Leave it running.

Do **not** run `--method utta` on CPU yet (20 MC passes × 85k images).

## 2. Colab CPU (if the VM still has data + checkpoint)

Same commands with `/content/...` paths. Slower than a local overnight run, but works without a GPU.

## 3. Free GPU alternative — Kaggle (best for Gate D later)

1. [kaggle.com/code](https://www.kaggle.com/code) → New notebook → GPU (T4/P100).
2. Settings → Accelerator = GPU, Internet = On.
3. ~30 GPU hours/week, resets weekly. No Colab quota.
4. Upload the 45 MB checkpoint as a Kaggle dataset, or copy from Drive.
5. Load Camelyon17 with:
   `datasets.load_dataset("wltjr1007/Camelyon17-WILDS")`
6. Use `notebooks/02_colab_tent_eata.ipynb` (same code).

## 4. Other GPUs (if Kaggle is tight)

| Option | Notes |
|---|---|
| Colab Pay As You Go | Works, costs compute units |
| Lightning.ai Studios | Free GPU hours on signup |
| RunPod / Vast.ai | ~$0.20–0.40/h T4 |
| Campus NVIDIA box | Best if you have one |

## Never

- Retrain source-only on CPU (~30 h)
- Adapt `smoke_s42_max20.pt`
- Start UTTA-Med until Tent + EATA JSON exist
