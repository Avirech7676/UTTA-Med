# Prediction Artifacts & Schema

Per-patch predictions for Camelyon17 test whole-slide images (Hospital 2, 85,054 patches across 10 slides) are generated during evaluation of each method across the 5 seeds.

### Parquet Schema for Prediction Dumps

Each prediction parquet (`seed{SEED}_{METHOD}.parquet`) contains:
- `image_id`: Unique patch identifier (e.g. `patch_camelyon17_slide_021_idx_00452`)
- `slide`: Whole-slide image identifier (slides 20–29)
- `label`: Ground-truth binary tumor label (0 = normal, 1 = tumor)
- `p_source`: Predicted tumor probability from frozen source model
- `p_after`: Predicted probability after test-time adaptation
- `uncertainty`: MC-dropout predictive variance $U(x) = \mathrm{Var}[\hat{p}(x)]$ ($N_{\mathrm{MC}}=20$)
- `prediction_before`: Binary prediction before TTA at threshold 0.5
- `prediction_after`: Binary prediction after TTA at threshold 0.5
- `adapted`: Boolean flag indicating whether patch was admitted by the gate ($U(x) < \tau$)

### Generation Command

To generate raw parquet dumps for any seed:
```bash
python scripts/adapt.py \
  --checkpoint checkpoints/camelyon17_resnet18_source_s42_BEST.pt \
  --method utta \
  --data-root "$CAMELYON17_ROOT" \
  --seed 42 \
  --save-predictions results/predictions/seed42_utta.parquet
```

All 5-seed aggregated metrics, per-seed metrics, and 1000-sample WSI bootstrap intervals computed from these predictions are archived in `results/metrics/` and `results/tables/`.
