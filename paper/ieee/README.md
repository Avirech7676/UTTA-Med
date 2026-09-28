# UTTA-Med IEEE / Overleaf pack (locked experiments)

Compile twice:

```bash
pdflatex utta_med.tex
pdflatex utta_med.tex
```

Overleaf: New project → Upload zip of this folder (`utta_med.tex` + `utta_med.bib` + `figs/`).
The `.tex` uses a self-contained `\begin{thebibliography}` so BibTeX is optional.
`utta_med.bib` is provided for LNCS / journal conversion.

## Frozen numbers — do not edit
Table 1 (mean±std) and Table 2 (per-seed AUROC) come from seeds `{42,123,2024,7,99}`.
WSI CIs: 10 hospital-2 slides, B=1000.

## Claims
Allowed: shift; Tent collapse 2/5 (slide-significant); gating cuts harm ~25×; F1 up 5/5; UTTA ≈ confidence; random is not enough; k≥5 collapses everyone.
Forbidden: “first uncertainty TTA”; “UTTA beats confidence”; dropping seeds 7/99; SOTA AUROC; clinical validity.

## Venue
ISBI / EMBC / MIDL / TTA workshop: this class.
MICCAI: swap to `llncs.cls`, keep tables.
TMI/MedIA: expand; you already have WSI CIs.
