# Phase 12 & 13 — Manuscript Final Review & Scientific Interpretation Lock

This document confirms the completion of **Phase 12 (Final Review Order Pass)** and **Phase 13 (Discussion, Limitations & Conclusion Lock)** for the UTTA-Med publication manuscript (`paper/ieee/utta_med.tex` and `paper/ieee/utta_med.bib`).

---

## 1. Acceptance Gate Audit Checklist

### 1.1 Scientific Evidence & Interpretation
- [x] **Every headline claim backed by frozen numerical evidence:**
  - Source AUROC: $0.930 \pm 0.009$, F1: $0.807 \pm 0.025$, ECE: $0.128 \pm 0.026$.
  - Tent AUROC: $0.698 \pm 0.324$, F1: $0.506 \pm 0.411$, ECE: $0.284 \pm 0.179$, Harm: $0.187 \pm 0.179$.
  - UTTA-Med AUROC: $0.935 \pm 0.022$, F1: $0.830 \pm 0.032$, ECE: $0.131 \pm 0.031$, Harm: $0.007 \pm 0.003$.
  - Confidence AUROC: $0.936 \pm 0.022$, F1: $0.831 \pm 0.032$, ECE: $0.130 \pm 0.030$, Harm: $0.007 \pm 0.003$.
  - Coverage-risk (50% keep): Error $16.60\% \to 3.60\%$ ($4.61\times$ reduction, $n=42{,}527$), AUROC $0.981 \pm 0.016$, ECE $0.036 \pm 0.029$.
  - Slide burden MAE: Source $0.100$, UTTA $0.097$, Tent $0.184$ (reaching $0.274$ on collapse seeds).
- [x] **Separation of F1 and AUROC statistical claims:**
  - $\Delta\mathrm{F1} = +0.0231$ is statistically significant ($95\%$ CI: $[+0.0104, +0.0358]$, $t=5.053$, $p=0.0072$, Cohen's $d=2.26$).
  - $\Delta\mathrm{AUROC} = +0.0058$ is non-significant ($p=0.4199$).
  - Manuscript explicitly refrains from claiming significant overall discrimination improvement.
- [x] **Confidence negative result explicitly acknowledged:**
  - Acknowledged in Abstract, Section IV-C, Section V (Discussion), Limitations, and Conclusion.
  - Transparently highlights that confidence gating achieves identical AUROC, slightly higher F1, and $5.7\times$ lower compute cost without stochastic passes.
- [x] **Strict absence of clinical validation claims:**
  - Uses "offline selective prediction error rate" and "whole-slide tumor burden concordance".
  - Completely purged "clinical error rate" and "clinical safety" from the manuscript.
- [x] **No cross-dataset generalization claims:**
  - Scope strictly restricted to Camelyon17-WILDS Center 2 across 5 seeds.
- [x] **No parameter-level causal claims for Tent collapse:**
  - Explicitly characterized as prediction-level probability suppression ($p < 0.01$ for $>89\%$), class-asymmetric sensitivity drop ($<3\%$), calibration distortion ($97\%$ forced into $[0.9, 1.0]$ with $51.4\%$ accuracy), and localization to high-tumor slides (28 and 29).
  - Explicitly states that internal BatchNorm parameter trajectories were not logged.
- [x] **Tent instability characterized as seed-dependent:**
  - Highlights collapse on 2 of 5 seeds (7 and 99), while remaining stable on 3 seeds (42, 123, 2024).
  - Emphasizes why 5 seeds are required to detect bimodal failure modes.
- [x] **Selective prediction separated from selective adaptation:**
  - Section IV-D explicitly delineates offline reject-option inference ranking from sample-level adaptation mask selection.

---

### 1.2 Citation & Bibliography Synchronization
- [x] **All 16 references independently verified:**
  - Documented in `docs/PHASE11_CITATION_VERIFICATION.md`.
- [x] **BibTeX metadata locked:**
  - Exact titles, authors, venues, years, volumes, DOIs, and arXiv identifiers in `paper/ieee/utta_med.bib`.
- [x] **Cite keys synchronized:**
  - Matching cite keys in `paper/ieee/utta_med.tex` (`dong2025certaintta`, `wu2026sictta`, `yu2025medsegtta`, `tsai2024ttsad`, `lou2025uadfm`, `noori2026histopathc`).
  - Added ECE metric reference `\cite{guo2017calibration}`.
- [x] **Accurate attribution:**
  - Corrected TT-SaD description to diffusion-based stain adaptation.

---

### 1.3 Engineering & Reproducibility
- [x] **Regression test suite:**
  - 57 passed, 2 skipped (100% pass rate).
- [x] **Leakage prevention verified:**
  - Zero target label access during adaptation.
  - Threshold $\tau$ calibrated exclusively on Hospital 1.
- [x] **Artifacts synchronized:**
  - Figures `figs/fig_slide_burden_estimation.png`, `figs/fig_coverage_risk_tradeoff.png`, and `figs/fig_tent_collapse_mechanism.png` embedded and referenced.

---

## 2. Locked Sections Text

### Abstract
> Deep learning models trained on histopathology patches from source hospitals exhibit sharp performance drops when deployed across clinical centers. Test-time adaptation (TTA) aims to mitigate distribution shift using unlabeled target streams, but unconstrained entropy minimization can produce catastrophic instability. In this five-seed reliability study on official Camelyon17-WILDS splits (source hospitals $\{0,3,4\} \rightarrow$ target hospital $2$), we investigate the mechanics of selective adaptation under a strictly leakage-controlled protocol: BatchNorm affine updates only, running statistics frozen, learning rate $10^{-5}$, Monte Carlo dropout ($N_{\mathrm{MC}}=20$), and gating thresholds frozen on hospital~$1$. On hospital~$2$, cross-hospital domain shift is real, with the source model achieving $0.930\pm0.009$ AUROC. Under identical hyperparameters, ungated Tent collapses catastrophically on $2/5$ seeds (mean AUROC $0.698\pm0.324$, harm rate $0.187\pm0.179$). In contrast, selective gating prevents collapse, cuts harmful adaptation to $0.007\pm0.003$, and improves F1 on all five seeds ($\Delta\mathrm{F1}=+0.0231$, $p=0.0072$). Random adaptation at matched coverage also collapses on the same two seeds, demonstrating that \emph{which} samples are adapted is what governs stability. On this binary classification task, MC-Dropout predictive variance does not outperform softmax confidence ($\Delta$~AUROC $<0.001$, confidence slightly higher F1), while incurring additional stochastic inference overhead. At matched coverage, selective gating reliably stabilizes test-time adaptation under the evaluated distribution shift, whereas unconstrained Tent exhibits severe seed-dependent instability.

### Discussion
> **Main Finding.** Selective adaptation substantially reduced harmful adaptation under the evaluated Camelyon17-WILDS cross-hospital shift and improved F1 relative to the frozen source model ($\Delta\mathrm{F1}=+0.0231$, 95\% CI: $[+0.0104, +0.0358]$, $t=5.053$, $p=0.0072$, Cohen's $d=2.26$). In contrast, overall discriminative improvement was modest and not statistically significant ($\Delta\mathrm{AUROC}=+0.0058$, $p=0.4199$). Thus, selective TTA functions primarily as a variance-reducing reliability stabilizer and positive-class alignment mechanism rather than dramatically expanding overall discriminative capacity.
>
> **Seed-Dependent Tent Instability.** Our empirical analysis indicates that Tent exhibits severe seed-dependent instability under the evaluated shift, rather than uniform degradation. While performing adequately on seeds 42, 123, and 2024 (AUROC $\ge 0.929$), Tent collapsed completely on seeds 7 and 99 (AUROC $\approx 0.34$, harm rate $>34\%$). This bimodal profile highlights why multi-seed evaluations are indispensable in medical TTA: an evaluation reporting only a single favorable seed would completely obscure this catastrophic failure mode.
>
> **Selectivity Versus Random Gating.** A key mechanistic question is whether selective adaptation succeeds simply because it reduces the number of gradient steps. Our matched-coverage random gating experiment directly refutes this: random gating at the same update frequency ($\approx 70\%$ coverage) collapsed on the exact same two seeds (seeds 7 and 99, AUROC $<0.38$). Therefore, reducing update frequency alone is insufficient; the identity and certainty of adapted samples govern test-time stability.
>
> **MC-Dropout Uncertainty Versus Softmax Confidence.** Importantly, our findings present an unambiguous negative result regarding MC-Dropout variance on binary histopathology classification. Softmax confidence achieves identical AUROC ($|\Delta\mathrm{AUROC}| < 0.001$) and slightly higher F1 ($0.831 \pm 0.032$ vs.\ $0.830 \pm 0.032$), while requiring only a single forward pass per patch rather than $N_{\mathrm{MC}}=20$ stochastic passes ($\approx 5.7\times$ lower computation cost). For binary classification, predictive entropy is monotonically coupled to softmax confidence, making confidence gating both mathematically equivalent and computationally superior. MC-Dropout variance does not demonstrate an accuracy advantage justifying its inference overhead in this setting.
>
> **Implications.** These results establish selective TTA as a viable reliability-oriented framework for cross-hospital distribution shift in computational pathology, preventing catastrophic collapse while suppressing harmful flips to $<0.8\%$. However, these findings remain specific to the evaluated distribution shift, leaving open whether the same behavior holds across different tissue types, modalities, multiclass tasks, architectures, and clinical workflows.

### Limitations
> We explicitly delineate the following scientific boundaries:
> 1. **Single Benchmark:** Primary findings are established exclusively on the official Camelyon17-WILDS benchmark splits.
> 2. **Binary Classification:** The task is restricted to central $32\times32$ lymph node metastasis classification; multi-class tissue grading and subtyping were not evaluated.
> 3. **Single Primary Architecture:** All adaptation experiments utilize a ResNet-18 benchmark backbone; transformer architectures and deeper convolutional backbones remain future work.
> 4. **Offline Retrospective Evaluation:** The selective prediction reject option and whole-slide concordance analyses are conducted retrospectively; prospective pathologist-in-the-loop workflow trials and clinical utility studies were not performed.
> 5. **Computational Overhead:** MC-Dropout requires $N_{\mathrm{MC}}=20$ stochastic forward passes, imposing an inference latency penalty.
> 6. **Competitive Baseline of Confidence:** MC-Dropout variance does not provide a demonstrated performance advantage over single-pass softmax confidence gating.
> 7. **Prediction-Level Diagnostic Scope:** Tent collapse is characterized through prediction probabilities, error bins, and slide localization; internal BatchNorm parameter trajectories were not logged to disk, precluding parameter-level causal claims.
>
> *Scope Boundary:* No claim of universal safety, cross-dataset generalization, or clinical workflow readiness is asserted.

### Conclusion
> This study evaluates selective test-time adaptation for cross-hospital histopathology classification under the official Camelyon17-WILDS protocol. Across five random seeds, ungated entropy-minimization TTA exhibited substantial seed-dependent instability, including catastrophic collapse on two seeds. Selective adaptation substantially reduced harmful prediction changes and improved F1 relative to the frozen source model, while matched-coverage random gating remained vulnerable to the same collapse behavior. However, MC-Dropout predictive variance did not outperform confidence-based gating on this binary task and incurred additional computational cost. These results support selective adaptation as a reliability-oriented approach for the evaluated distribution shift, while motivating validation across additional datasets, tasks, architectures, and prospective clinical settings.

### Reproducibility Statement
> This study strictly adheres to the official Camelyon17-WILDS protocol (\texttt{wltjr1007/Camelyon17-WILDS}). Source models are trained on hospitals $\{0,3,4\}$, validation hospital~$1$ is used exclusively for hyperparameter freezing and threshold calibration ($\tau$ at 70th percentile of uncertainty), and target hospital~$2$ (85,054 patches, 10 WSIs) is evaluated strictly once with ground-truth labels completely isolated and unavailable during adaptation. Experiments are conducted across five fixed random seeds $\{42, 123, 2024, 7, 99\}$ using ResNet-18 with ImageNet initialization, Adam optimizer ($\mathrm{lr}=10^{-5}$ for TTA, $\mathrm{lr}=10^{-4}$ for source training), batch size 64, BatchNorm affine updates only, frozen running statistics, $N_{\mathrm{MC}}=20$ stochastic dropout passes, and an empty-batch gradient skip guard. The repository includes complete configurations, evaluation pipelines, and regression test suites (57 passing unit/regression tests, 2 expected skips for offline raw data). All numerical values in this manuscript match the frozen metric dumps in \texttt{results/metrics/}.
