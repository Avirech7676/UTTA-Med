# Kaggle GPU Execution Risk & Mitigation Report

**Project:** UTTA-Med Project 2 — GPU Automation  
**Status:** ACTIVE RISK MITIGATION  

---

## 1. Identified Technical Bottlenecks & Failure Modes

1. **Kaggle 6.0-Hour Container Timeout:**
   * Single Kaggle batch sessions enforce a hard cutoff at $21,600\text{ seconds}$ (6 hours).
   * At $\sim 45\text{ minutes per epoch}$, 5 epochs for 1 seed requires $3.75\text{ hours}$.
   * Training all 5 seeds sequentially in one session would take $\sim 18.75\text{ hours}$ and **will trigger container termination**.
   * **Mitigation:** Seed 42 will finish within the 6-hour window and serialize its checkpoint. Subsequent seeds (123, 2024, 7, 99) will be executed across separate modular Kaggle batch sessions.
2. **Rejection of CPU Smoke-Test Artifacts:**
   * Checkpoint `camelyon17_densenet121_source_s42_BEST.pt` (30.8 MB) created on CPU from 128 synthetic samples (`val_auroc = 0.5582`) is formally rejected.
   * Only genuine GPU checkpoints trained on the 302,436 real patches are admitted.
