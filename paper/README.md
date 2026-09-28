# UTTA-Med IEEE draft

Compile (needs `IEEEtran.cls`, usually from TeX Live):

```bash
pdflatex utta_med.tex
pdflatex utta_med.tex
```

Overleaf: new project → upload this folder (`utta_med.tex` + `figs/`).

Venue notes:
- **ISBI / EMBC / IEEE conference:** this class is correct (`\documentclass[conference]{IEEEtran}`).
- **MICCAI:** switch to LNCS (`llncs.cls`), keep the same text and tables.
- **TMI / MedIA journal:** expand related work and add WSI-level CIs when per-patch dumps exist.

Do not edit Table 1 numbers. They are frozen from seeds {42, 123, 2024, 7, 99}.
