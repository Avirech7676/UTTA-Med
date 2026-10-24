# Final Issues and Corrections Master Document

**Project:** UTTA-Med Project 2  
**Audit Standard:** Master Prompt Part H & Part J  

---

## Itemized Issues and Authoritative Corrections

### Issue I01: Analytical Projection F1 = 0.8723
* **Problem:** Previous report presented an analytical projection as an empirical result.
* **Correction:** Formally rejected the 128-sample CPU checkpoint; committed to zero fabrication and genuine GPU execution.

### Issue I06: Latency Unit Ambiguity
* **Problem:** Ambiguity between per-batch and per-patch timing (136.29 ms).
* **Correction:** Audited standardized B=32 batch timing ($6.77\text{ ms}$ forward, $7.68\text{ ms}$ adapt, $136.29\text{ ms}$ MC-dropout) in `CORRECTED_GPU_LATENCY.csv`.

### Issue I07: Confidence Gating Coverage Discrepancy
* **Problem:** Conflicting reports of 58.1% vs 83.74% coverage.
* **Correction:** Reconciled that 83.74% represents full-cohort ($N=85,054$) sample admission, while 56.2% represents the balanced diagnostic subset.

### Issue I08: Harm Rate Conflation
* **Problem:** Conflation of harm definitions ($H/N$ vs $H/k$).
* **Correction:** Established 5-tier harm taxonomy separating admitted error updates from post-adaptation prediction flips.

### Issue I09: Detection Cohort Mismatch
* **Problem:** Multiple overlapping manifests (118 vs 131 vs 117 mitoses).
* **Correction:** Strictly segregated historical paper benchmark (118 mitoses) from advanced ablations (131 and 117 mitoses).

### Issue I10: D8 vs D9 Redundancy
* **Problem:** Identical reported metrics for D8 and D9.
* **Correction:** Audited raw predictions; proved exact bounding box invariance; certified D8 as the parsimonious champion under Occam's razor.

### Issue I11: Statistical Discrepancy (t=10.67, df=2)
* **Problem:** Reporting inconsistency with p=0.0041.
* **Correction:** Recalculated two-sided $p=0.0087$, one-sided $p=0.0043$, and case-clustered bootstrap $p=0.0187$.
