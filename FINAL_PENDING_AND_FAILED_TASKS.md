# Final Pending, Blocked, and Failed Tasks Audit

**Project:** UTTA-Med Project 2  
**Audit Date:** October 24, 2026  

---

## 1. Pending Tasks (Awaiting Active GPU Execution)

| Task ID | Phase | Task Name | Current State | Reason Pending / Blocking Dependency |
|:---:|:---|:---|:---|:---|
| **117** | Phase 8 | Complete genuine training for Seed 42 | IN PROGRESS | Actively computing Epoch 1 on Kaggle 2x Tesla T4 GPU (~52m elapsed). |
| **118** | Phase 8 | Save & verify genuine checkpoint for Seed 42 | PENDING | Awaits completion of 5 epochs for Seed 42. |
| **119** | Phase 8 | Complete genuine training for Seed 123 | QUEUED | Scheduled after Seed 42 artifact retrieval. |
| **120** | Phase 8 | Complete genuine training for Seed 2024 | QUEUED | Scheduled after Seed 123. |
| **121** | Phase 8 | Complete genuine training for Seed 7 | QUEUED | Scheduled after Seed 2024. |
| **122** | Phase 8 | Complete genuine training for Seed 99 | QUEUED | Scheduled after Seed 7. |
| **123–126** | Phase 8 | Save & verify checkpoints for Seeds 123, 2024, 7, 99 | PENDING | Dependent on Tasks 119–122. |
| **127–138** | Phase 8 | Checkpoint integrity & loadability audits | PENDING | Dependent on physical checkpoint creation. |
| **139–141** | Phase 9 | Evaluate Arms E0, E1, E2 on Center 2 | PENDING | Dependent on completed DenseNet-121 checkpoints. |
| **143–148** | Phase 9 | Save prediction dumps & compute empirical statistics | PENDING | Dependent on Tasks 139–141. |
| **149–160** | Phase 9 | Attribution separation & paired statistical tests | PENDING | Dependent on 5-seed prediction arrays. |
| **161** | Phase 10 | Evaluate target goals (F1 >= 0.85, AUROC >= 0.945) | PENDING | Awaiting empirical results from Phase 9. |
| **175** | Phase 12 | Verify improved model artifacts in registry | PENDING | Dependent on Phase 8 completion. |

---

## 2. Failed Tasks Audit
* **Count:** **0 Failed Tasks.**
* **Explanation:** No experiments have crashed or failed. The single CPU smoke-test checkpoint was an unvalidated artifact that was audited and formally **rejected** rather than failed.

---

## 3. Blocked Tasks Audit
* **Count:** **0 Blocked Tasks.**
* **Status:** Kaggle GPU quota is active (30.0h available), credentials are authenticated, and execution is progressing normally.
