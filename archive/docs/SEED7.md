# Seed 7 — run next (do not retune)

Frozen recipe (same as 42 / 123 / 2024):
- BN running stats frozen
- TTA Adam lr = 1e-5
- N_MC = 20
- τ = 70th percentile of **this seed's** hospital-1 U
- EATA e_margin = 0.4
- Test hospital 2 once

Notebook: `kaggle_seed7_full.ipynb`

Download from Output:
- `camelyon17_resnet18_source_s7_BEST.pt`
- `camelyon17_resnet18_full_s7.json`

Then seed 99. Do not change HPs.
