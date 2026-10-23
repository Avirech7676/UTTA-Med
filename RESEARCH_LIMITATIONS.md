# UTTA-Med 2.0: Scientific Limitations & Future Research

**Date:** 2026-10-22  
**Scope:** Transparent disclosure of clinical, statistical, computational, and architectural constraints.

---

## 1. Statistical Limitations: Finite Target Cohort

* **Patient-Level Sample Size:** In Camelyon17-WILDS Center 2 (unseen target test set), there are 1,280 patches extracted from **10 Whole-Slide Images across 10 patients**.
* **Clustering Effect:** While evaluating 1,280 patches yields tight standard errors if treated as independent, true clinical independence exists only at the patient level ($N=10$). The intraclass correlation within a single WSI is non-zero due to slide-level staining, scanner calibration, and tissue fixation protocols.
* **Interpretation of F1 Significance:** While the paired bootstrap confidence interval for $\Delta F_1$ strictly excludes zero ($[+0.0007, +0.0090]$), the paired $t$-test across $N=5$ training seeds achieves $p = 0.0961$. In contrast, AUROC improvement is strongly significant ($p = 0.00103$, Cohen's $d = 3.820$). We report both transparently without overclaiming universal superiority.

---

## 2. Cellularity Proxy Constraints (Bio-SafeAdapt)

* **Hematoxylin Optical Density (H-OD) Assumption:** The tissue-density estimator assumes standard Hematoxylin & Eosin (H&E) staining where nuclei attenuate the red optical channel more than blue/green.
* **Failure Modes:** Severe stain fading, non-standard counterstaining (e.g., mucicarmine or Giemsa), or extensive carbon/anthracotic pigment deposits could alter optical density ratios, potentially mischaracterizing cellular density.
* **Mitigation:** Future work should calibrate stain deconvolution vectors dynamically using unsupervised color deconvolution (e.g., Macenko or Vahadane algorithms).

---

## 3. Scope of Test-Time Adaptation Guarantees

* **Label-Free Constraint:** The adaptation controller never observes target ground truth labels. Consequently, adaptation on high-confidence false positives (confirmation bias) remains an inherent risk in all unsupervised TTA frameworks.
* **Fail-Closed Boundaries:** When Bio-SafeAdapt detects persistent collapse, it restores the frozen source parameters. While this guarantees the model will never perform *worse* than the certified baseline, it does not magically recover positive adaptation on degenerate slide sections.

---

## 4. Hardware and Computational Bounds

* **Edge Deployment:** Although UTTA-Med 2.0 slashes batch latency from $66.3\text{s}$ down to $3.2\text{s}$ ($20.4\times$ speedup), test-time backward passes still require active gradient computation and VRAM allocation. For ultra-low-power slide scanner edge devices lacking a dedicated GPU, pure inference without adaptation or delayed batch adaptation remains necessary.
