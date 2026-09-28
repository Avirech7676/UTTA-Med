# Gate C diagnosis (updated after BEST-ckpt run)

## Gate B — SIGNED OFF

Checkpoint: `camelyon17_resnet18_source_s42_BEST.pt`  
Epoch 3, seed 42, Tesla T4. Loaded OOD AUROC **matches** file metadata (0.9793).

| Split | AUROC | AUPRC | F1 | ECE | Role |
|---|---:|---:|---:|---:|---|
| ID val (0/3/4) | 0.9992 | 0.9992 | 0.9882 | 0.002 | in-domain |
| OOD val (1) | 0.9793 | 0.9798 | 0.9203 | 0.050 | HP / τ only |
| **Target (2)** | **0.9355** | **0.9446** | **0.8350** | **0.099** | final eval |

Shift (ID − target): AUROC drop **0.064**, F1 drop **0.153**.  
Target sensitivity 0.75 vs specificity 0.95 — the source model is conservative on hospital 2 (misses tumors). This is the RQ1 result.

Do **not** quote the earlier Colab smoke or the epoch-8 overwrite numbers in the paper.

## Gate C attempt 2 — Tent/EATA on the BEST ckpt (not official yet)

Lr selected on OOD val from {1e-4, 2.5e-4, 1e-3} → **1e-4** (least bad).  
**All three lrs collapsed on val.** Test at 1e-4 also collapsed.

| Setting | Split | AUROC after | Harm | Coverage |
|---|---|---:|---:|---:|
| Tent 1e-4 | OOD val | 0.567 | 0.378 | 1.00 |
| Tent 2.5e-4 | OOD val | 0.559 | 0.385 | 1.00 |
| Tent 1e-3 | OOD val | 0.529 | 0.403 | 1.00 |
| Tent 1e-4 | test | 0.567 | 0.359 | 1.00 |
| EATA 1e-4, e=0.4 | test | 0.560 | 0.363 | 0.928 |

This is **not** a strawman on the wrong checkpoint — source test AUROC is 0.935.  
It **is** still too early to freeze Tent: 1e-4 is the bottom of the grid, so we have not shown that Tent cannot work. Reviewers will ask for smaller lr and frozen BN running stats.

Likely cause: continual Tent in `BN.train()` updates **batch running mean/var** (buffers, not just γ/β). One pass of 85k target patches poisons those stats. Entropy min then drives predictions to 0/1.

## Next (no retrain)

Run `kaggle_gate_c_tent_salvage.ipynb`:

1. Load BEST.pt only.
2. On OOD val, two BN modes × four lrs:
   - `bn_train` = original Tent (batch stats)
   - `bn_freeze_stats` = affine γ/β only, running stats frozen (stricter “BN-affine-only”)
   - lr ∈ {1e-5, 3e-5, 5e-5, 1e-4}
3. Pick the pair with **highest OOD-val AUROC**, provided it does not fall >0.02 below source val AUROC. If every pair collapses, Tent collapse is the official finding.
4. EATA e_margin ∈ {0.2, 0.4, 0.55} at that lr/mode, still on val.
5. One test pass.

Do not start UTTA-Med until this salvage finishes. If Tent still collapses, UTTA-Med is tested as a *collapse-prevention* method — that is still a valid paper, but Tent must be the best Tent we can get on val, not a missed lr.
