# Submission Cover Letter: Bio-SafeAdapt & Digital Pathology TTA Audit

**Manuscript Title:** Biologically Informed Fail-Closed Test-Time Adaptation for Robust Digital Pathology Under Scanner Domain Shift  
**Author:** Avinash Reddy Cheerapareddy  
**Affiliation:** Department of Computer Science and Engineering, Computational Pathology and Medical Imaging Laboratory  
**Corresponding Contact:** Avinash Reddy Cheerapareddy (avinashreddy.crp@gmail.com)  
**Target Venue:** IEEE Transactions on Medical Imaging (TMI) / Medical Image Analysis (MedIA)  

---

**Dear Editor-in-Chief and Associate Editors,**

We are pleased to submit our original research manuscript entitled *"Biologically Informed Fail-Closed Test-Time Adaptation for Robust Digital Pathology Under Scanner Domain Shift"* for consideration in IEEE Transactions on Medical Imaging.

### Clinical Problem & Scientific Motivation
Deep neural networks deployed for digital pathology diagnosis frequently suffer severe performance degradation when transferred across clinical institutions due to shifts in slide preparation, staining protocols, and whole-slide image (WSI) scanners. While unsupervised Test-Time Adaptation (TTA) methods such as entropy minimization (Tent) attempt to mitigate distribution shift online using unlabeled target patient data, they exhibit catastrophic instability: under a standard BatchNorm-affine protocol on the multicenter Camelyon17-WILDS benchmark, unconstrained Tent collapses completely on 2 of 5 random seeds ($F_1 < 0.10$, with AUROC dropping from $0.93 \to 0.34$).

### Key Empirical Findings & Methodological Breakthroughs
This manuscript presents an end-to-end empirical, theoretical, and biological investigation across two clinical histopathology tasks:

1. **Track A (Multicenter Metastasis Classification — $N=85,054$ target patches, 5 random seeds):**
   - **Empirical Demystification of Gating Complexity:** We evaluate seven adaptation controllers across five random seeds. We show that complex multi-signal feature gating (C7) yields $F_1 = 0.8318 \pm 0.0314$, but simple Softmax Confidence Gating (C1) achieves statistically indistinguishable accuracy ($F_1 = 0.8310 \pm 0.0321, p=0.693$) while executing $24.8\%$ faster in forward gating ($6.77$\,ms vs $8.45$\,ms per batch of 32) and $1.98\times$ faster in full adaptation update ($7.68$\,ms vs $15.20$\,ms).
   - **Rigorous Matched-Coverage Safety Audit:** Under strict matched update budgets ($k=719$ accepted updates per seed across seven coverage budgets), Confidence Gating admits significantly fewer harmful updates than C7 ($55.0 \pm 2.8$ vs $63.2 \pm 3.1$ errors, paired $t=10.25, p=0.0005$, Cohen's $d_z = 4.58$).
   - **Theoretical & Biological Derivation:** We prove mathematically that on single-logit binary classification, Monte Carlo Dropout variance and max-softmax confidence induce near-identical sample rankings ($\text{Spearman } \rho = -0.9917$). Furthermore, full-cohort discordance analysis reveals the biological etiology of MC-Dropout underperformance: **asymmetric tumor starvation** ($92.89\%$ of excluded updates are true tumor margins), causing stroma-dominated gradient updates that collapse classifier sensitivity on low-prevalence micrometastasis slides.
   - **Slide-Level Staging Invariance:** Top-$K$ patch max-pooling preserves whole-slide macro AUROC ($0.9308 \to 0.9300$) and maintains $100\%$ concordance with ground-truth pN-stage metastasis classifications across all 10 target slides.

2. **Track B (Dense Mitotic Figure Detection — MIDOG 2021, 150 WSIs across 3 optical scanners):**
   - **Discovery of Acellular Stroma Safeguard Failure:** Existing fail-closed controllers (Original SafeAdapt) mistake normal acellular stroma (which naturally contains zero mitoses) for catastrophic detector collapse, raising $2.67$ false silence alarms per seed and forcing premature permanent fallback to frozen source mode ($F_1 = 0.4630$).
   - **Proposed Bio-SafeAdapt Architecture:** By extracting Hematoxylin Optical Density ($H\text{-OD}$) via Ruifrok-Johnston color deconvolution, Bio-SafeAdapt conditions silence alarms on verified cellularity ($H\text{-OD} \ge 0.12$), completely eliminating false alarms in acellular tissue ($0.00$ vs $2.67$).
   - **Statistically Verified Positive Transfer:** Bio-SafeAdapt integrates elastic parameter anchoring ($\lambda = 0.01$) and episodic case-boundary resets, achieving $F_1 = 0.4732 \pm 0.0023$ on held-out Scanner~3---an authentic incremental gain of $+0.0102$\,$F_1$ over SafeAdapt (paired $t=10.67, p=0.0087$, Cohen's $d_z = 6.16$) and $+0.0147$\,$F_1$ over Frozen Source ($0.4585, p < 0.0001, t=17.15$). 10,000 paired case-clustered bootstrap resamples across 50 whole-slide images confirm spatial adaptation significance ($p = 0.0187$).
   - **Parsimonious Architecture (D8):** We demonstrate that configuration D8 (Cellularity + Anchoring + Resets) achieves identical performance to D9 at lower latency ($77.0$\,ms vs $77.2$\,ms) and lower VRAM ($1{,}960$\,MB vs $1{,}970$\,MB).

### Compliance, Open Reproducibility & Integrity
- The complete source code, automated regression test suite (73 passing unit tests), raw prediction arrays ($425,270$ target predictions), and frozen model weights (SHA-256 verified) are organized in full compliance with IEEE reproducibility requirements.
- The research was conducted strictly on public, de-identified benchmark datasets (Camelyon17-WILDS and MIDOG 2021).
- All negative results, outlier failure modes (Slide 29), and methodological boundaries are fully and transparently disclosed.

Thank you for your consideration of our work.

Sincerely,  
**Avinash Reddy Cheerapareddy**  
Department of Computer Science and Engineering  
Computational Pathology and Medical Imaging Laboratory  
Corresponding Author
