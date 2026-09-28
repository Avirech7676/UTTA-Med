# UTTA-Med: Grad-CAM Interpretability Audit (Five Categories)

This document formalizes the interpretability audit of the ResNet-18 model under test-time adaptation on Camelyon17 target hospital 2 patches, organized into the five predefined diagnostic categories.

---

## 1. Audit Categorization Framework

Interpretability in medical TTA requires validating that adaptation alters model attention toward clinically relevant histopathological structures rather than spurious staining artifacts:

| Category | Definition | Representative Asset | Diagnostic Significance |
| :--- | :--- | :--- | :--- |
| **1. Correct (High-Confidence Concordant)** | Patches correctly classified with high certainty ($p > 0.90$, low $U(x)$). | `paper/figures/gradcam/source_TP.png`<br>`paper/figures/gradcam/source_TN.png` | Confirms model attends to true metastatic tumor nests (cellular atypia, hyperchromatic nuclei) or uniform normal stroma. |
| **2. Incorrect (High-Shift Errors)** | Patches misclassified by source model with false positive/negative diagnoses. | `paper/figures/gradcam/source_FP.png`<br>`paper/figures/gradcam/source_FN.png` | Identifies stain-induced false alarms (benign germinal centers, histiocytes) and missed micro-metastases. |
| **3. High Uncertainty (Boundary Samples)** | Patches near the decision threshold ($p \approx 0.50$, high $U(x) > \tau$). | `paper/figures/gradcam/source_highU_wrong.png`<br>`paper/figures/gradcam/frozen_highU_wrong.png` | Demonstrates that the MC-Dropout predictive variance gate successfully captures ambiguous histological boundaries and excludes them from adaptation. |
| **4. Wrong $\to$ Correct (Adapted Benefit)** | Samples where selective TTA restored a missed diagnosis without label supervision. | `paper/figures/gradcam/compare_FN.png`<br>`frozen_lowU_correct.png` | Shows how BN affine re-centering sharpens attention onto subtle tumor features previously missed due to stain variation. |
| **5. Correct $\to$ Wrong (Harm Avoidance)** | Samples flipped into error by unconstrained Tent ($18.7\%$ harm rate), but protected by UTTA ($0.7\%$ harm rate). | `results/statistical/five_seed_final/fig_harm_per_seed.png`<br>`ADAPT_STEPS_GRADCAM_SIGNOFF.md` | Validates that selective gating blocks entropy minimization from forcing false certainty onto normal lymphoid aggregates. |

---

## 2. Quantitative Manifest Reference

Patch coordinates, ground truth labels ($y \in \{0, 1\}$), model predictive probabilities ($p$), and MC-Dropout uncertainty values ($U(x)$) are cataloged in [`paper/figures/gradcam/gradcam_manifest.json`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/paper/figures/gradcam/gradcam_manifest.json).

```json
{
  "scanned": 512,
  "positives": 271,
  "ckpt": "camelyon17_resnet18_source_s42_BEST.pt",
  "layer": "layer4.1.conv2",
  "signoff": "Frozen-source uncertainty calibration (0 harm events in CAM set)"
}
```

---

## 3. Methodological Claim & Scientific Scope

> **Important Scientific Scope**:
> Grad-CAM visual heatmaps serve strictly as an **interpretability audit** illustrating qualitative shifts in convolutional feature activations across source and adapted states. They demonstrate model sensitivity to tissue morphology versus background stroma, but do not constitute standalone clinical diagnostic evidence.
