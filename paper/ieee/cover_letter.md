# Submission Cover Letter: UTTA-Med / SafeAdapt

**Manuscript Title:** Selective Test-Time Adaptation for Reliable Cross-Hospital Histopathology Classification: A Five-Seed Study on Camelyon17-WILDS  
**Author:** Avinash Reddy Cheerapareddy  
**Affiliation:** Department of Computer Science and Engineering  
**Corresponding Contact:** Avinash Reddy Cheerapareddy  
**Target Venue:** IEEE International Symposium on Biomedical Imaging (ISBI) / IEEE Transactions on Medical Imaging (TMI)  

---

**Dear Editor-in-Chief and Associate Editors,**

We are pleased to submit our original research manuscript entitled *"Selective Test-Time Adaptation for Reliable Cross-Hospital Histopathology Classification: A Five-Seed Study on Camelyon17-WILDS"* for consideration.

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

4. **Transparent Evaluation of External Target Domain & Limitations:**  
   We report an isolated, pre-adaptation streaming evaluation on an external target hospital (Hospital 2, 10 WSIs, 1,280 patches). We show that while safeguards prevented gross collapse and contained parameter drift ($\Delta \le 4.01 \times 10^{-5} \ll 0.05$), adaptation yielded slight negative transfer relative to the frozen source model (Macro AUROC $0.9308 \to 0.9300$, Pooled F1 $0.7840 \to 0.7822$), driven by confirmation bias on challenging outlier slides (e.g., Slide 29). We transparently discuss these findings to caution against unmonitored clinical deployment of unsupervised TTA in computational pathology.

### Compliance, Data Availability & Code Reproducibility
- The manuscript is an original work and is not under consideration for publication elsewhere.
- All authors have reviewed and approved the manuscript.
- Secondary analysis was conducted strictly on public, de-identified benchmark datasets (Camelyon17-WILDS).
- The complete source code, automated regression test suite (120 passed unit tests), configuration files, and frozen checkpoint weights (canonical SHA-256: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`) are packaged in full compliance with IEEE reproducibility requirements.

Thank you for your time and consideration of our manuscript.

Sincerely,  
**Avinash Reddy Cheerapareddy**  
Department of Computer Science and Engineering  
Corresponding Author
