# Paste into Antigravity ONLY after Gate B-FULL JSON exists with cuda: true

You are continuing UTTA-Med. Master Plan v4 is frozen.

Gate C — Tent + EATA on the FROZEN source checkpoint
camelyon17_resnet18_source_s42.pt from a FULL GPU run (not smoke_s42_max20.pt).

Rules:
- torch.cuda.is_available() must be True. Else stop.
- Continual adaptation, one pass over target, shuffle=False, steps=1.
- Update BatchNorm affine (γ, β) only. Identical scope for Tent and EATA.
- Tent: entropy minimization, lr=1e-3.
- EATA: entropy filter e_margin=0.4 (binary entropy, max≈0.693), same lr, skip batch if none pass the filter.
- Target labels: metrics only, never for the update.
- Record coverage (Tent=1.0), harm_rate, correction_rate, ECE, Brier, AUROC, AUPRC, F1.
- Do not tune on test. Optional lr/e_margin peek on OOD val (center 1) only — if you peek, log it.

Preferred: notebooks/02_colab_tent_eata.ipynb in the SAME Colab runtime that trained the source model (HF dataset already cached).

Write results/metrics/camelyon17_resnet18_tent_eata_s42.json
Do not start UTTA-Med / gates until Tent and EATA both finish.
Do not fabricate metrics.
