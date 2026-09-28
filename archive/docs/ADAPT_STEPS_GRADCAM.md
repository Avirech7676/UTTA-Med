# Adaptation-step sweep + post-TTA Grad-CAM (seed 42)

No retrain. Add Input `camelyon17_resnet18_source_s42_BEST.pt`.

## Part 1 — inner steps `{1,2,5,10}` on hospital 1

One pass over OOD val. `k` = SGD steps **per batch**, not extra epochs.
Official recipe stays **k=1**.

Watch whether Tent/UTTA AUROC/harm stay flat or collapse as `k` grows.

## Part 2 — same-patch Grad-CAM

Columns: patch | source | Tent k=1 | UTTA k=1  
Rows: TP, TN, FP, FN, high-U wrong, low-U correct.

Audit only, not clinical validation.

## Send back

- printed step-sweep table
- `camelyon17_adapt_steps_gradcam_s42.json`
- `figures/fig_adapt_steps_val_s42.png`
- `gradcam_post_tta/compare_*.png`
