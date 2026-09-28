# Research Questions — UTTA-Med

**Frozen protocol — do not change after target-test results are seen.**

## Primary Research Goal
Determine whether uncertainty-aware sample selection (MC-Dropout predictive variance gate) can make test-time adaptation safer and more reliable under real cross-hospital distribution shift on Camelyon17-WILDS.

## Research Questions

**RQ1**  
How much does source-only performance degrade when a model trained on source hospitals is evaluated on an unseen hospital?

**RQ2**  
Can standard TTA methods (Tent, EATA) recover performance under cross-hospital distribution shift?

**RQ3**  
Does MC-Dropout predictive uncertainty correlate with target-domain prediction errors?

**RQ4**  
Can uncertainty-based sample selection reduce harmful TTA updates (Correct → Wrong)?

**RQ5**  
Does UTTA-Med improve calibration (ECE, Brier) and reliability in addition to classification performance?

**RQ6**  
Does MC-Dropout uncertainty gating provide an advantage over simpler gating strategies (Random, Confidence, Entropy) under matched coverage?

**RQ7**  
Can Grad-CAM reveal useful differences in model attention before and after adaptation (correct, incorrect, high-uncertainty, corrected, and harmfully adapted samples)?

## Governing Principle
Results may be positive, mixed, or negative. All outcomes are scientifically valid when experiments are rigorous, leakage-free, and honestly reported.
