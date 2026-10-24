# Verified Test-Time Adaptation Protocols & Telemetry

**Project:** UTTA-Med Project 2 — Method Comparison  
**Methods:** Arm E0 (Frozen), Arm E1 (Confidence Gating), Arm E2 (Improved EMA UTTA-Med)  

---

## 1. Experimental Evaluation Protocol

* **Evaluation Cohort:** Camelyon17-WILDS Center 2 ($85,054$ patches, 10 WSIs, Patients 040–059).
* **Source Anchor Rule:** Within each seed, all three arms (E0, E1, E2) initialize from the **identical source checkpoint**.
* **Online Streaming:** Patches evaluated sequentially by slide; whole-slide boundary resets clear adapted normalization buffers between patients.
* **Admission Condition:** $\max(p, 1-p) \ge \tau_{\text{conf}} = 0.85$.
