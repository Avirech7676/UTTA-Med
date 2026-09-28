# Adaptation steps + post-TTA Grad-CAM — seed 42

## Part 1 — VALID. Freeze k=1.

Inner SGD steps per batch, one pass, hospital 1.

| k | Tent AUROC | EATA AUROC | UTTA AUROC | Harm (all ~) |
|---|----------:|----------:|----------:|-------------:|
| 1 | 0.977 | 0.977 | 0.977 | 0.02 |
| 2 | 0.972 | 0.970 | 0.956 | 0.04–0.07 |
| 5 | **0.375** | **0.361** | **0.304** | **0.52** |
| 10 | **0.335** | **0.314** | **0.280** | **0.52** |

Official TTA remains **k=1**. Gating does **not** prevent collapse when you take 5–10 inner steps. UTTA coverage also opens (0.76 → 0.98) as U collapses to ~0 on a degenerate model.

Paper: this figure is the justification for k=1.

## Part 2 — Tent column OK; UTTA column INVALID

Test Tent k=1 AUROC **0.929** matches Gate D. Use source vs Tent CAMs.

Test UTTA k=1 AUROC **0.324** (recall 0.044, p≈0 everywhere) does **not** match official Gate D UTTA (**0.951**).

Cause: this notebook scored MC-Dropout **on the adapting model**. Official UTTA-Med uses **frozen source U** to choose who adapts. Online U under hospital-2 shift lets the gate open and entropy min drives all logits negative (blank CAMs = 1-logit artifact at p=0).

Do **not** put the UTTA `p=0.00 ERR` galleries in the paper as “what UTTA does.”

`utta_harm` on the 11 low-U tumors is that collapse, not the official method.

## What to keep

- `fig_adapt_steps_val_s42.png`
- Source vs Tent columns of compare_*.png
- Official UTTA numbers = Gate D JSON (frozen-source gate)
- Source-only Figure 7 from Grad-CAM v2

Optional later: one Grad-CAM rerun with frozen-source U. Not required for the headline paper.
