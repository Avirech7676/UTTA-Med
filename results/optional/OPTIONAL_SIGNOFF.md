# Optional GPU track — official status

**Not Table 1.** Camelyon17 ResNet-18 five-seed remains the primary result.

| ID | Experiment | Status | Key numbers | Cite? |
|----|------------|--------|-------------|-------|
| X1 | ResNet-50 seed 42 **source** | **Done** | Target AUROC **0.920** (weaker than R18 **0.935**). OOD 0.949 | One sentence supplement only |
| X1 | ResNet-50 Tent / UTTA | **Rejected** | NaNs / harm bug | **Do not cite** |
| X5 | ResNet-50 seeds 123, 2024, 7, 99 | **Cancelled** | — | **Do not run** |
| X2 | PCam = Camelyon16 patches | **Done** | Source AUROC **0.934**. UTTA **0.942**, harm **0.017**, F1 still below source | Supplement only |
| X3 | PathMNIST + TTA under C | **Done** | Clean acc **0.905**, AUROC **0.985**. Severe C drops acc; UTTA avoids harm | Supplement: TTA does **not** fix severe C |
| X4 | MIDOG mitosis/impostor LODO | **Done** | Source AUROC **0.8575**, AUPRC **0.7929**, UTTA harm **0.0192** (lowest harm) | Supplement: Scanner-shift classification proxy |
| P1 | NCT-CRC-HE (100K → 7K) | **Done** | Source Acc **90.42%**, F1 **0.891**, AUROC **0.9897**. UTTA harm **0.0%** (4 corrections) | Supplement: Multi-class histology generalization |
| H3 | MIDOG Faster R-CNN Target TTA (R3a) | **Certified Superior** | Arm E (SafeAdapt) F1 **0.4512** vs Source **0.4446** ($\Delta = +0.0077$, 95% CI $[+0.0009, +0.0152]$) | Stage H3.2-R3a: Case-clustered TOST superiority |
| — | Full CAMELYON16 WSI, MIDOG++ detection, PCam as Table 1 | **Skip** | — | — |

## Where files live

See `results/optional/README.md`.
