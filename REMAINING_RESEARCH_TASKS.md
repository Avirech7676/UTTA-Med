# REMAINING RESEARCH TASKS & METHODOLOGICAL ROADMAP

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Auditor:** Senior Software Architect & Computational Pathology Researcher  

---

## 1. REPOSITORY STATUS: VERIFIED & FROZEN

The active UTTA-Med repository is in a publication-grade, fully reproducible state:
1. **Repository Structure Cleaned:** Root clutter eliminated; external archive active at `C:/Users/avina/Documents/UTTA_MED_RESEARCH_ARCHIVE`.
2. **Architectural Integrity Preserved:** ResNet-18 (single-logit BCE) and MIDOG Faster R-CNN (torchvision backbone) verified.
3. **Historical Data Untouched:** Checkpoint hashes and authentic bootstrap predictions ($N=85,054$) match reference manifests.
4. **All Regression Tests Passing:** $73$ passed, $5$ skipped, $0$ failed.

---

## 2. SCIENTIFIC CONCLUSIONS FROM EMPIRICAL EVIDENCE

1. **Head-Only MC-Dropout is Fundamentally Limited in Binary Pathology:** Applying dropout exclusively after spatial pooling creates near-perfect colinearity between predictive variance and confidence. It cannot identify out-of-distribution textures or histological artifacts that fool the deterministic network.
2. **Deterministic Confidence is the Optimal Gate for Head Adaptation:** Deterministic margin ($|p - 0.5|$) achieves superior tumor recall ($0.8506$ vs $0.8495$) and fewer harmful flips ($1,142$ vs $1,194$) than MC-Dropout.
3. **The Core Contribution is Safeguarded Selective Adaptation:** The paper's primary impact lies in demonstrating that unconstrained adaptation (Tent, EATA) catastrophically collapses in digital pathology, whereas selective gating (Confidence Gating or UTTA-Med) with fail-closed safeguards ensures stable clinical performance under domain shift.

---

## 3. RECOMMENDED FUTURE RESEARCH TASKS (POST-SUBMISSION)

For future research extensions beyond the current frozen manuscript charter:

1. **Feature-Level Epistemic Estimators (Backbone Dropout):**
   - *Hypothesis:* Applying MC-Dropout across residual blocks (or test-time augmentations like RandStainNA) would capture spatial/chromatic uncertainty that is decoupled from output margin.
   - *Requirement:* Requires backbone retraining with spatial dropout layers.
2. **Slide-Level Aggregation Context:**
   - *Hypothesis:* Utilizing Whole Slide Image (WSI) context (e.g., surrounding tissue cellularity) to modulate patch admission thresholds rather than isolated patch confidence.
3. **Conformal Prediction Calibration:**
   - *Hypothesis:* Calibrating non-conformity scores on OOD validation centers (Center 1) to guarantee finite-sample coverage guarantees on target centers (Center 2).
