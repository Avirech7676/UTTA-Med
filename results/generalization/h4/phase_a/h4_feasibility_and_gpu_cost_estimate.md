# UTTA-Med -- Stage H4: Computational Feasibility and GPU Cost Estimate

**Evaluation Protocol:** 5 Adaptation Arms (A-E) x 3 Certified Seeds (42, 7, 123)  
**Evaluated Target Population:** 1,100 Spatially Distributed 512x512 Tiles across 50 Scanner 3 Cases  
**Compute Platform:** Kaggle Tesla T4 GPU (14.56 GB VRAM)  
**Analysis Date:** 2026-10-16  

---

## 1. Inference Budget Breakdown

| Evaluation Arm | Adaptation Method | Backpropagation | Est. Latency / Patch | Patches / Seed | Est. Runtime / Seed | 3-Seed Runtime |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Arm A** | Frozen Baseline | None (Inference Only) | ~32 ms | 1,100 | ~35.2 s | ~1.8 min |
| **Arm B** | Conventional TTA | Entropy Backprop (Affine) | ~58 ms | 1,100 | ~63.8 s | ~3.2 min |
| **Arm C** | Bounded TTA | Entropy + L_inf Projection | ~60 ms | 1,100 | ~66.0 s | ~3.3 min |
| **Arm D** | Monitored TTA | Entropy + Drift Rollback | ~60 ms | 1,100 | ~66.0 s | ~3.3 min |
| **Arm E** | SafeAdapt (Full) | Source-Anchored Gating + Dual Forward | ~74 ms | 1,100 | ~81.4 s | ~4.1 min |
| **Total** | **All 5 Arms** | -- | -- | **5,500** | **~308.4 s (~5.1 min)** | **~15.7 min** |

---

## 2. Resource & Quota Feasibility

- **Total Patch Forward Passes:** 1,100 x 5 x 3 = 16,500 patch inferences.
- **Estimated End-to-End GPU Execution Time:** **~15.7 minutes** (including data loading and metric exports).
- **Kaggle Tesla T4 Weekly Quota:** 30.0 hours.
- **Quota Utilization:** ~0.26 hours (**0.87% of weekly allocation**).
- **VRAM Headroom:** Peak batch-size=1 memory consumption is ~1.8 GB on a 14.56 GB Tesla T4 (87.6% headroom). Out-of-memory risk is negligible.

---

## 3. Potential 44-Patch Expansion Option Feasibility

If an expanded evaluation (N = 2,200 patches, 44 patches/case) were selected:
- Estimated GPU Execution Time: ~31.4 minutes (1.7% of weekly quota).
- **Recommendation:** The primary locked 1,100-window budget (18.02 mm2 tissue) captures **118 mitotic figures** (compared to 117 in H3.2-R3b), providing ample statistical power for the paired bootstrap while maintaining a lean footprint.
