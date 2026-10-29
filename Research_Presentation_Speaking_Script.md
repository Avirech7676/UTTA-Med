# Research Presentation Speaking Script

**Manuscript Title:** *SafeAdapt: Provably Bounded, Fail-Closed Test-Time Adaptation with Optimal Transport and Biologically Conditioned Safeguards for Digital Pathology*  
**Presenter:** Avinash Reddy Cheerapareddy  
**Co-Author:** Gunnam Sanjana  
**Institution:** Department of Computer Science & Engineering, SRM University-AP  
**Target Duration:** 14–15 Minutes (Paced at ~130 words per minute)

---

## 1. Slide-by-Slide Speaking Script

### Slide 1: Title Slide
* **Exact Spoken Script:**  
  "Good morning, respected members of the committee, faculty, and examiners. My name is Avinash Reddy Cheerapareddy, and today I am presenting my research work entitled: *SafeAdapt: Provably Bounded, Fail-Closed Test-Time Adaptation with Optimal Transport and Biologically Conditioned Safeguards for Digital Pathology*. The core objective of this work is to solve a fundamental vulnerability in clinical medical AI: how to safely adapt deep neural networks to new hospitals in real time during clinical scanning without risking catastrophic diagnostic collapse."
* **Timing:** 0:45
* **Key Point to Emphasize:** Establish an authoritative, safety-first tone immediately.

---

### Slide 2: Research Problem — Clinical Domain Shift
* **Exact Spoken Script:**  
  "When deep learning models are trained in computational pathology, they achieve high accuracy on tissue slides from the training hospital. However, as shown on this slide, when deployed to external medical centers, variations in tissue preparation, chemical staining protocols, and optical scanner hardware cause severe distribution shifts. Unlabeled target models suffer severe degradation, dropping diagnostic accuracy precisely when clinicians need reliable automated assistance."
* **Timing:** 0:45
* **Key Point to Emphasize:** Stain and scanner shifts cause deep models to fail in new hospitals.

---

### Slide 3: Clinical Urgency & High Stakes
* **Exact Spoken Script:**  
  "Why can we not simply deploy standard adaptation algorithms? In digital pathology, diagnostic stakes are absolute. An algorithmic false negative in an axillary lymph node examination means missed metastatic breast cancer, leading to delayed systemic chemotherapy. If an autonomous model updates its parameters online without supervision, incorrect predictions create a positive feedback loop of confirmation bias. In medicine, we cannot accept unmonitored silent failures."
* **Timing:** 0:45
* **Key Point to Emphasize:** Computational pathology cannot tolerate silent algorithmic corruption.

---

### Slide 4: Existing Methods — Vision TTA
* **Exact Spoken Script:**  
  "To address distribution shifts during inference, computer vision researchers developed Test-Time Adaptation, or TTA. Leading methods like Tent update Batch Normalization layers by minimizing prediction entropy. Others, like EATA, filter samples using entropy thresholds, while SAR uses sharpness-aware optimization to avoid sharp minima, and CoTTA continually restores stochastic weights. These algorithms perform well on natural image corruptions like fog or blur."
* **Timing:** 0:45
* **Key Point to Emphasize:** Existing TTA was built for natural images, not clinical pathology.

---

### Slide 5: The Fundamental Research Gap
* **Exact Spoken Script:**  
  "However, all existing TTA methods operate on an unverified foundational premise: that low-entropy target predictions are predominantly correct. Under severe histochemical stain shift, this premise completely collapses. In our multi-seed benchmarks on the CAMELYON17 consortium dataset, we discovered that Tent catastrophically collapses on 40% of random seeds, driving $F_1$ scores down to 0.05. Existing regularizers only throttle update frequency; they cannot restore corrupted feature geometry."
* **Timing:** 0:50
* **Key Point to Emphasize:** Tent collapses on 40% of seeds because confident predictions are wrong.

---

### Slide 6: Central Research Question
* **Exact Spoken Script:**  
  "This brings us to our central research question: *Can geometry-preserving transport alignment and fail-closed runtime interlocks contain catastrophic TTA failure modes while achieving statistically verified positive adaptation?* In short: can we guarantee that test-time adaptation never performs worse than the source model while still extracting domain transfer gains?"
* **Timing:** 0:35
* **Key Point to Emphasize:** Frame the core question around certified runtime safety.

---

### Slide 7: Research Objectives
* **Exact Spoken Script:**  
  "To answer this question, we established four rigorous technical objectives: First, prove mathematically why scalar uncertainty gating fails. Second, design an Optimal Transport mechanism that preserves class geometry under severe stain drift. Third, establish provable parameter-space trust regions that bound maximum output logit shifts. Fourth, eliminate cross-patient error accumulation through deterministic episodic resets."
* **Timing:** 0:45
* **Key Point to Emphasize:** Walk through the four clear, structured objectives.

---

### Slide 8: Key Contributions
* **Exact Spoken Script:**  
  "Our work delivers four principal contributions. In Proposition 1, we formally prove that MC-Dropout variance is rank-equivalent to softmax confidence, proving why scalar gating fails. Second, we introduce SafeAdapt-OT, an exact-marginal Optimal Transport formulation. Third, in Theorem 1, we prove that an $L_\infty \le 0.05$ trust region bounds output probability shifts to under 0.0460. Finally, we implement sub-frame 1.8 millisecond atomic GPU rollbacks and verify our framework across multi-center pathology benchmarks."
* **Timing:** 1:00
* **Key Point to Emphasize:** Highlight the theoretical proofs alongside practical engineering.

---

### Slide 9: SafeAdapt Framework Overview
* **Exact Spoken Script:**  
  "Here is the overall architecture of SafeAdapt. During clinical inference, Whole-Slide Images stream patch batches into the network. All convolutional feature extraction weights remain strictly frozen to preserve learned semantics. Adaptation is restricted entirely to the affine scale and shift parameters of the normalization layers. As features emerge, they pass through our Optimal Transport alignment and biological gating heads, enveloped by a parameter-space trust region and monitored by dual asynchronous GPU-CPU ring buffers."
* **Timing:** 0:55
* **Key Point to Emphasize:** Frozen convolutional filters preserve semantics; only BN affine adapts.

---

### Slide 10: Component 1 — Optimal Transport Prototype Alignment
* **Exact Spoken Script:**  
  "Component 1 is Geometry-Preserving Optimal Transport. Under severe stain shifts, standard entropy minimization pulls all target patches into a single dominant class attractor. To prevent this, SafeAdapt-OT solves an Entropic Wasserstein-2 transport problem against certified source prototypes using real-time Sinkhorn-Knopp iterations. Crucially, we enforce the exact marginal constraint: $\mathbf{Q}^T \mathbf{1}_B = \mathbf{r}$, where $\mathbf{r}$ is the certified source class prior. This mathematically prohibits the model from assigning all patches to one class."
* **Timing:** 1:00
* **Key Point to Emphasize:** The exact marginal constraint $\mathbf{Q}^T \mathbf{1}_B = \mathbf{r}$ mathematically prevents collapse.

---

### Slide 11: Component 2 — Trust Regions & Atomic Rollbacks
* **Exact Spoken Script:**  
  "Component 2 provides our safety guarantees: Parameter-Space Trust Regions and Atomic Rollback Mechanics. In Theorem 1, we prove that projecting affine parameters within an $L_\infty \le 0.05$ box bounds output probability displacement to $\Delta p \le 0.0460$. Furthermore, SafeAdapt maintains active GPU memory snapshots. If gradient norms spike or consecutive alarms fire, the system executes an atomic rollback in 1.8 milliseconds, restoring clean weights before the next batch is processed. If alarms persist, it locks to certified source parameters."
* **Timing:** 1:00
* **Key Point to Emphasize:** Theorem 1 guarantees $\Delta p \le 0.0460$, backed by 1.8 ms GPU rollback.

---

### Slide 12: Experimental Setup & Consortium Datasets
* **Exact Spoken Script:**  
  "We conducted our empirical validation across two major multi-center clinical consortium benchmarks. For patch metastasis classification, we used CAMELYON17-WILDS, comprising 30 whole-slide images across 5 medical centers. For dense object detection, we evaluated on MIDOG 2021, covering 1,100 high-power fields across 4 distinct optical scanners. All experiments were repeated across 5 random seeds under a frozen BatchNorm running statistics protocol, benchmarking against 8 leading adaptation methods."
* **Timing:** 0:50
* **Key Point to Emphasize:** Multi-center, multi-scanner, multi-seed evaluation with 8 baselines.

---

### Slide 13: Main Results — CAMELYON17 Classification
* **Exact Spoken Script:**  
  "Here are our primary classification results on CAMELYON17. On Slide 29—the most severe histochemical outlier in the consortium—the source baseline achieved 0.8592 AUROC. Standard Tent collapsed completely to 0.0521, and recent Sharpness-Aware Adaptation dropped to 0.6482. In contrast, SafeAdapt-OT reversed negative transfer, elevating performance to 0.9324 AUROC. Across the entire 5-center cohort, SafeAdapt achieved a macro-AUROC of 0.9495, maintaining a certified 0.0% collapse rate across all seeds."
* **Timing:** 1:10
* **Key Point to Emphasize:** SafeAdapt reverses Slide 29 collapse ($0.0521 \to 0.9324$) with 0.0% collapse rate.

---

### Slide 14: Dense Mitosis Detection on MIDOG 2021
* **Exact Spoken Script:**  
  "Turning to dense mitosis detection on MIDOG 2021: optical scanner shifts cause standard detectors to confuse rare mitotic cells with background tissue. Furthermore, acellular stroma contains no cells, which causes standard TTA to trigger false alarms. Our Centroid Margin Regularization directly separates mitotic from non-mitotic prototypes, boosting detection $F_1$ score to 0.5172 on Scanner 3—a statistically significant improvement of +6.87% over baseline with $p < 0.001$."
* **Timing:** 0:50
* **Key Point to Emphasize:** Centroid Margin Regularization yields $+6.87\%$ $F_1$ gain on cross-scanner detection.

---

### Slide 15: Ablation Studies & Permutation Invariance
* **Exact Spoken Script:**  
  "Our ablation experiments reveal critical insights into streaming adaptation. First, executing multiple gradient steps per batch ($k \ge 2$) causes rapid feature drift, degrading AUROC to 0.7420 at $k=5$; single-step updates ($k=1$) are strictly necessary. Second, removing the trust region allows parameters to drift outside the safe basin. Finally, our slide arrival permutation tests confirmed that episodic resets reduce cross-patient variance to mathematically zero ($\sigma^2_{\text{perm}} = 0.000000$), ensuring complete patient isolation."
* **Timing:** 1:00
* **Key Point to Emphasize:** Single-step updates and episodic resets are essential for stability.

---

### Slide 16: Qualitative Evidence — Feature Geometry
* **Exact Spoken Script:**  
  "This slide provides qualitative UMAP visual proof of feature representations on outlier Slide 29. On the left, the source model shows moderate separation. In the center, unconstrained Tent collapses both tumor and normal representations into a single degenerate clump, losing all discriminative ability. On the right, SafeAdapt-OT restores clean bimodal geometry, reconstructing well-separated tumor and normal clusters that match the true underlying biology."
* **Timing:** 0:50
* **Key Point to Emphasize:** UMAP shows Tent collapsing into one clump, while SafeAdapt restores 2 clusters.

---

### Slide 17: Stress Testing, Staging & Hardware Viability
* **Exact Spoken Script:**  
  "To prove clinical deployability, we stress-tested SafeAdapt across extreme regimes. In single-patch streaming where batch size $B=1$, an online FIFO memory buffer maintained 0.9482 AUROC. On purely benign lymph node slides containing 0.0% tumor cells, SafeAdapt contained false positive rates to 1.1%, compared to 42.6% for Tent. Crucially, inference forward latency was just 35.6 milliseconds per patch with a 474 MB VRAM footprint—over 8 times lighter than generative diffusion stain transfer."
* **Timing:** 1:05
* **Key Point to Emphasize:** Works at $B=1$, controls false alarms on 100% benign slides, runs in 35.6 ms.

---

### Slide 18: Limitations & Translational Roadmap
* **Exact Spoken Script:**  
  "We acknowledge two main limitations: First, our Optimal Transport formulation assumes known class prior capacity; in future work, we are incorporating a virtual 'dustbin' prototype to absorb open-set non-neoplastic inflammatory mimics. Second, adaptation is confined to affine normalization parameters. For future clinical translation, we are preparing Stage-2 prospective non-inferiority trials across 140 patients with direct DICOMweb WADO-RS PACS integration triggered by slide accession numbers."
* **Timing:** 0:50
* **Key Point to Emphasize:** Honest limitations (class priors) and clear roadmap (open-set dustbin, PACS).

---

### Slide 19: Conclusion
* **Exact Spoken Script:**  
  "In conclusion, SafeAdapt demonstrates that test-time adaptation in clinical medicine cannot rely on unconstrained entropy minimization. By combining Optimal Transport class-prior alignment, provable parameter-space trust regions, sub-frame atomic rollbacks, and episodic resets, SafeAdapt establishes a certified fail-closed medical interlock. It achieves state-of-the-art diagnostic gains where viable and guarantees zero harm where shifts are intractable. Thank you, and I look forward to your questions."
* **Timing:** 0:50
* **Key Point to Emphasize:** Deliver a memorable concluding sentence on certified fail-closed safety.

---

## 2. Short Versions for Quick Presentations

### 5-Minute Short Version
> "Good morning. Deep neural networks deployed in digital pathology frequently fail when encountering tissue slides from external hospitals due to differences in chemical staining and optical scanners. While Test-Time Adaptation aims to adjust models online using unlabeled target images, popular methods like Tent rely on entropy minimization. In our multi-seed benchmarks, we found that Tent catastrophically collapses on 40% of random seeds because confident predictions under stain shift are frequently wrong, creating a destructive feedback loop.
> 
> To solve this, we developed **SafeAdapt**, a fail-closed test-time adaptation framework built on four pillars: First, we proved in Proposition 1 that Monte Carlo Dropout variance is rank-equivalent to confidence on binary classification heads ($\rho = -0.9917$), proving that uncertainty filtering cannot stop representation collapse. Second, we formulated adaptation as an Entropic Optimal Transport problem via Sinkhorn-Knopp iterations, enforcing an exact class-prior marginal constraint ($\mathbf{Q}^T \mathbf{1}_B = \mathbf{r}$) that mathematically prohibits single-class collapse. Third, in Theorem 1, we proved that projecting parameters within an $L_\infty \le 0.05$ trust region bounds maximum output probability displacement to $\Delta p \le 0.0460$, backed by sub-frame 1.8 millisecond GPU state rollbacks. Fourth, deterministic episodic resets guarantee complete patient diagnostic independence ($\sigma^2_{\text{perm}} = 0.000000$).
> 
> Empirically, on CAMELYON17 outlier Slide 29 where Tent collapses to 0.0521 AUROC, SafeAdapt reverses negative transfer to reach 0.9324 AUROC, elevating cohort macro-AUROC to 0.9495 with certified 0.0% collapse across all seeds. On MIDOG 2021 mitosis detection, our Centroid Margin Regularization improves $F_1$ score to 0.5172 (+6.87%, $p < 0.001$). SafeAdapt executes in 35.6 milliseconds on under 500 MB VRAM, establishing a certified, fail-closed medical interlock ready for clinical reading room integration. Thank you."

### 3-Minute Short Version
> "Deep learning models in digital pathology suffer severe performance degradation when exposed to stain variations and optical scanner shifts across different hospitals. Standard test-time adaptation methods like Tent attempt to adapt online via entropy minimization, but we show that Tent collapses on 40% of random seeds, driving $F_1$ scores down to 0.05 due to self-reinforcing confirmation bias.
> 
> We introduce **SafeAdapt**, a mathematically bounded, fail-closed adaptation framework. SafeAdapt replaces heuristic entropy minimization with Entropic Optimal Transport, enforcing exact marginal prior constraints that mathematically prevent single-class attractor collapse. To provide strict safety guarantees, Theorem 1 proves that an $L_\infty \le 0.05$ parameter trust region bounds output probability shifts to under 0.0460. If anomalies occur, dual-buffer GPU rollbacks restore clean model parameters in just 1.8 milliseconds, while per-slide episodic resets ensure zero cross-patient error propagation.
> 
> Across 30 multi-center CAMELYON17 slides and 1,100 cross-scanner MIDOG patches, SafeAdapt achieves certified 0.0% collapse, reversing negative transfer on outlier Slide 29 from 0.0521 to 0.9324 AUROC and improving mitotic detection $F_1$ by +6.87%. Running in 35.6 milliseconds with under 500 megabytes of VRAM, SafeAdapt provides the mathematical guarantees required to deploy adaptive AI safely in clinical pathology laboratories. Thank you."

### 1-Minute Pitch
> "Deploying diagnostic AI across hospitals is fundamentally hindered by histochemical stain and scanner variations. While test-time adaptation attempts to update models during inference, standard entropy minimization collapses on 40% of random seeds, misdiagnosing malignant tumors as benign. 
> 
> In this research, we introduce **SafeAdapt**, the first provably bounded, fail-closed adaptation framework for digital pathology. SafeAdapt uses Optimal Transport with exact class-prior marginal constraints to mathematically eliminate single-class collapse. By bounding parameter drift within an $L_\infty \le 0.05$ trust region, Theorem 1 guarantees that output probabilities cannot shift by more than 0.0460, backed by sub-frame 1.8 millisecond GPU state rollbacks and deterministic per-slide resets. 
> 
> On CAMELYON17 outlier Slide 29, SafeAdapt reverses catastrophic failure, reaching 0.9324 AUROC with certified zero percent collapse across all seeds. SafeAdapt transforms fragile adaptation into a certified medical safety interlock."

### 30-Second Elevator Pitch
> "Clinical pathology AI frequently breaks when transferred between hospital scanners due to stain shifts. Standard online adaptation makes this worse—collapsing on 40% of random seeds because confident predictions are wrong. We created **SafeAdapt**, which combines Optimal Transport to preserve class geometry with a mathematical trust region that provably limits parameter drift. SafeAdapt completely eliminates collapse across multi-center consortium benchmarks, reverses negative transfer on clinical outliers, and executes in 35 milliseconds with zero patient cross-contamination."
