# UTTA-Med IEEE draft

Overleaf: New project → Upload this `ieee/` folder → compile `utta_med.tex` **twice**.

## Frozen claims (do not edit Table 1)

- Source target AUROC **0.930 ± 0.009**
- Tent collapses on seeds **7 and 99** — keep those seeds
- Gating harm **0.187 → 0.007**; F1 up on **5/5**
- **UTTA-Med ≈ confidence**
- Camelyon17-C and PathMNIST are **supplement / appendix**, not Table 1

## You still type

1. Author names, affiliation, email
2. Venue (ISBI / EMBC / MIDL = this `IEEEtran` class; MICCAI = LNCS)

## Do not

- Drop seeds 7/99
- Claim UTTA beats confidence
- Claim first uncertainty-aware medical TTA
- Quote Tent mean AUROC 0.70 without “collapsed 2/5”
- Mix PathMNIST numbers into the Camelyon17 table
