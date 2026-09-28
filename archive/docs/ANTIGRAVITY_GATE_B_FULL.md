# Paste this into Antigravity after you have a CUDA GPU (Colab / Kaggle / lab)

You are continuing UTTA-Med. Master Plan v4 is frozen.

## Hard stop from Gate B smoke

The file checkpoints/camelyon17_resnet18_source_s42.pt from --max-batches 20
is a SMOKE checkpoint. Do not adapt it. Do not put those metrics in any paper table.
Tag it smoke: rename to checkpoints/smoke_s42_max20.pt if it still exists.

Local PC has no NVIDIA GPU. Full training must run on Colab/Kaggle/CUDA.

## Gate B-FULL (source-only, seed 42)

1. Confirm torch.cuda.is_available() is True. If False, stop and say so.
2. Train ResNet-18, 1-logit, BCEWithLogitsLoss, dropout 0.5, ImageNet pretrained.
3. Train on centers {0,3,4}. Early-stop on OOD val center {1}. Evaluate test center {2} only after the checkpoint is frozen.
4. AMP (torch.amp) on. batch_size 64 or 128. Adam 1e-4, wd 1e-4, epochs 20, patience 5, seed 42.
5. Windows local: num_workers 0. Colab/Linux: num_workers 2.
6. Save checkpoints/camelyon17_resnet18_source_s42.pt
7. Write results/metrics/camelyon17_resnet18_source_s42.json with:
   id_val, val_ood, test_target, shift (id_val − test) for AUROC, AUPRC, F1, Acc, ECE, Brier
   plus n_epochs_trained, device, amp, git-free config dump.
8. Do not run Tent/EATA/UTTA until this JSON exists from a FULL run (not max-batches).

Preferred entry points:
- Colab: notebooks/01_colab_source_train.ipynb
- CLI: python scripts/train_baseline.py --seed 42 --epochs 20 --num-workers 2

If Colab disk is tight, use HuggingFace dataset wltjr1007/Camelyon17-WILDS.
Validation parquet is MIXED: center 1 = ood_val, centers 0/3/4 = id_val. Split by center.

After the run, zip src, scripts, configs, results/metrics (no .venv, no parquet, no .pt if >50MB).
Include the metrics JSON. That is the Gate B-FULL review package.

Do not fabricate metrics. Do not tune on test.
