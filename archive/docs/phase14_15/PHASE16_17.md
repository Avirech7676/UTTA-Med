# Phases 16–17 (documents last)

## Phase 16 — Paper

IEEE draft already exists: `paper/ieee/utta_med.tex` + `utta_med.bib` + `UTTA-Med-overleaf.zip`.

Fill from **Phase 15 tables** (do not invent numbers). Remaining writing:

- Author block / venue
- Table 8 blank until ResNet-50 JSON, or omit B2/B9
- Limitations already listed (10 WSIs, one target hospital, UTTA≈confidence)

Do **not** start Phase 16 edits that change Table 1.

## Phase 17 — Release hygiene (partial)

Done: README frozen protocol, `.gitignore`, leakage YAML, 5-seed CSVs.

Still: delete dummy `experiments/.../metrics.json` auroc=0.0 if present; do not commit `.pt` or parquet; rewrite local Windows paths to `CAMELYON17_ROOT`; no HF tokens.

Public code zip should be **code + CSVs + paper**, checkpoints on Drive.
