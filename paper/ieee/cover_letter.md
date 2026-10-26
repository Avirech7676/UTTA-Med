# Submission Cover Letter: SafeAdapt

**Manuscript Title:** SafeAdapt: Geometry-Preserving Optimal Transport and Fail-Closed Safeguards for Test-Time Adaptation in Digital Pathology  
**Author:** Avinash Reddy Cheerapareddy  
**Affiliation:** Department of Computer Science and Engineering  
**Corresponding Contact:** Avinash Reddy Cheerapareddy  
**Target Venue:** IEEE International Symposium on Biomedical Imaging (ISBI) / IEEE Transactions on Medical Imaging (TMI)  

---

**Dear Editor-in-Chief and Associate Editors,**

We are pleased to submit our original research manuscript entitled *"SafeAdapt: Geometry-Preserving Optimal Transport and Fail-Closed Safeguards for Test-Time Adaptation in Digital Pathology"* for consideration.

### Scientific Context & Problem Addressed
Deep neural networks deployed for digital pathology diagnosis frequently suffer performance degradation when transferred across clinical institutions due to shifts in tissue preparation, staining protocols, and whole-slide image (WSI) scanners. Unsupervised Test-Time Adaptation (TTA) methods such as entropy minimization (Tent) update model parameters online using unlabeled target batches. However, as documented in our rigorous five-seed study, unconstrained entropy minimization exhibits severe instability: under a fair BatchNorm-affine protocol on Camelyon17-WILDS (lymph node metastasis classification), Tent collapses completely on 2 of 5 random seeds (AUROC drops from $0.93 \to 0.34$, with sensitivity dropping below $3\%$).

### Key Empirical Findings & Methodological Contributions
This manuscript presents an empirical investigation of selective test-time adaptation, failure modes, and runtime safeguarding under cross-hospital shift:

1. **Selective Uncertainty Gating Cuts Harmful Adaptation ~25-Fold:**  
   We introduce hard selective gating using an uncertainty threshold $\tau$ calibrated exclusively on a held-out institution (Hospital 1). Across five random initialization seeds, gating admits low-uncertainty patches to BatchNorm-affine updates, reducing the harmful adaptation rate from $18.7\% \pm 17.9\%$ (Tent) to $0.7\% \pm 0.3\%$ and yielding a consistent paired F1 improvement on all five seeds ($\Delta\text{F1} = +0.023$, $95\%$ CI $[+0.010, +0.036]$, $p = 0.0072$).
   
2. **Matched-Coverage Controls & Empirical Equivalence:**  
   By evaluating matched-coverage random gating controls (which still collapse on seeds 7 and 99), we demonstrate that performance gains arise specifically from *which* samples are adapted, rather than simply taking fewer optimization steps. Furthermore, on single-logit binary classification, Monte Carlo dropout predictive variance and max-softmax confidence induce near-identical patch rankings (Spearman $\rho = -0.9917$). Deterministic confidence gating matches MC-dropout accuracy while eliminating a $3.5\times$ computational latency penalty.

3. **Runtime Safeguards & Slide-Boundary Invariance (SafeAdapt):**  
   We evaluate runtime safeguarding featuring explicit parameter displacement bounds ($\|\theta_t - \theta_0\|_\infty \le 0.05$), non-finite gradient interception, atomic rollback, and whole-slide boundary resets. These safeguards prevent numerical divergence and eliminate inter-slide error accumulation across independent biological specimens.

4. **Reversing Negative Transfer via Optimal Transport & CMR Gating:**  
   We demonstrate that while standard entropy minimization exhibits negative transfer on histochemical outliers (e.g., Slide 29 dropping to $0.0521$ AUROC), our proposed Entropic Optimal Transport (OT-TTA) and Beer-Lambert optical density consistency mathematically prevent majority-class attractor collapse ($Q^T \mathbf{1}_B = \mathbf{r}$), elevating Slide 29 to $\mathbf{0.9324}$ and achieving a statistically significant positive adaptation gain across the 5-seed Hospital 2 cohort ($\mathbf{0.9495} \pm 0.0134$ Macro AUROC, $p = 0.0312$, paired Wilcoxon test; ECE cut by $43.2\%$). Furthermore, on MIDOG 2021 mitosis detection, Cellularity-Mitosis Ratio (CMR) gating eliminates all 14 false silence alarms on acellular stroma, yielding stable continuous adaptation ($F_1 = \mathbf{0.5172} \pm 0.0012$, $p < 0.001$).

### Compliance, Data Availability & Code Reproducibility
- The manuscript is an original work and is not under consideration for publication elsewhere.
- All authors have reviewed and approved the manuscript.
- Secondary analysis was conducted strictly on public, de-identified benchmark datasets (Camelyon17-WILDS and MIDOG 2021).
- The complete source code, automated regression test suite (148 passed unit tests), configuration files, verified Kaggle Tesla T4 GPU training outputs, and adapted checkpoint weights (SHA-256: `185cb4c7215ad10a82051c6bde14b22ca9c732463b2111542892b748221a9a1b`) are packaged in full compliance with IEEE reproducibility requirements.

Thank you for your time and consideration of our manuscript.

Sincerely,  
**Avinash Reddy Cheerapareddy**  
Department of Computer Science and Engineering  
Corresponding Author
