# Figure 7 — Grad-CAM audit (signed off)

Source ResNet-18, seed 42, hospital 2. 512 shuffled patches.
Counts: TP 215 · TN 226 · FP 15 · FN 56.

**Not clinical validation.** 96×96, last conv block, 1-logit Grad-CAM.

| Panel | Use in paper | One-line reading |
|---|---|---|
| `source_lowU_correct.png` | Left of Fig 7 | Confident correct: tumor CAM on nuclei; normal CAM blank (fat/stroma) |
| `source_highU_wrong.png` | Right of Fig 7 | Errors with $p$ near 0.5; CAM on edges — samples the gate skips |
| `source_TP.png` | Supplement | CAM on cellular tumor-like regions |
| `source_TN.png` | Supplement | Fat/stroma; blank maps = strongly negative logit |
| `source_FP.png` | Supplement | Dense lymphoid-like tissue; some $p>0.96$ (gate would accept) |
| `source_FN.png` | Supplement | Missed tumor; mix of uncertain ($p\sim0.4$) and confident ($p\sim0.01$) misses |

Caption:

> Figure 7. Grad-CAM on the seed-42 source model, unseen hospital 2 (audit, not clinical validation).
> Low-uncertainty correct predictions attend to dense nuclei (tumor) or produce near-blank maps on fat/stroma (normal).
> High-uncertainty errors sit near $p=0.5$ with edge-heavy maps — the population excluded from UTTA-Med updates.
> Confident false positives on lymphocyte-rich patches remain a residual failure mode: a hard gate cannot block a wrong sample that the model is sure about.
