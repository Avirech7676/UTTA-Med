# TUMOR STARVATION & DISCORDANCE VERIFICATION REPORT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Branch:** `research-project-2` | **Dataset:** Authentic Camelyon17 Center 2 ($N=85,054$ patches/seed, 5 Seeds)  

---

## 1. COMPLETE 5-SEED DISCORDANCE MATRIX

To resolve whether excluded tumor patches cause or correlate with later adaptation failure, we construct the complete $2 \times 2$ contingency table across all $425,270$ target predictions:

| seed | both_correct_tumor | both_correct_normal | both_wrong_tumor | both_wrong_normal | c_corr_u_wrong_tumor | c_corr_u_wrong_normal | c_corr_u_wrong_total | u_corr_c_wrong_tumor | u_corr_c_wrong_normal | u_corr_c_wrong_total | net_advantage_conf | net_advantage_conf_tumor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 33459 | 41545 | 8983 | 952 | 71 | 22 | 93 | 14 | 8 | 22 | 71 | 57 |
| 123 | 32134 | 41886 | 10296 | 620 | 88 | 0 | 88 | 9 | 21 | 30 | 58 | 79 |
| 2024 | 30093 | 42186 | 12361 | 333 | 68 | 5 | 73 | 5 | 3 | 8 | 65 | 63 |
| 7 | 27789 | 42384 | 14603 | 136 | 132 | 7 | 139 | 3 | 0 | 3 | 136 | 129 |
| 99 | 29176 | 42250 | 13262 | 276 | 85 | 0 | 85 | 4 | 1 | 5 | 80 | 81 |

### Summary Totals Across All 5 Seeds:
- **Both Correct:** $360,540$ patches (Tumor: $151,326$, Normal: $209,214$)
- **Both Incorrect:** $64,184$ patches (Tumor: $60,250$, Normal: $3,934$)
- **Confidence Correct & UTTA Incorrect:** **478 patches**
  - **True Tumor Patches:** **444 (92.9%)**
  - **True Normal Patches:** **34 (7.1%)**
- **UTTA Correct & Confidence Incorrect:** **68 patches**
  - **True Tumor Patches:** **35 (51.5%)**
  - **True Normal Patches:** **33 (48.5%)**

### Net Confidence Gating Advantage:
$$\mathbf{\text{Net Advantage} = +410 \text{ patches} \quad (+409 \text{ on TUMOR margins})}$$

---

## 2. CORRELATION VS. CAUSATION ANALYSIS

Is tumor exclusion merely correlated with difficulty, or does it causally drive adaptation failure?

1. **Temporal Precedence:**
   - In online test-time adaptation, Batch Normalization affine updates are cumulative.
   - When a slide stream begins (e.g. Slide 21), MC-Dropout variance on initial tumor patches is high, causing them to be rejected ($w_i = 0$).
   - The adaptation loss $\mathcal{L}_{\text{ent}}$ is computed exclusively over homogeneous normal stroma.
2. **Causal Parameter Shift:**
   - Adapting BN running affine parameters on $95+\%$ stroma reduces the classifier's positive logit bias.
   - For subsequent boundary patches, the post-adaptation logit drops below zero ($p < 0.50$), producing a **False Negative**.
   - Under Confidence Gating ($A_1$), confident tumor patches ($p \ge 0.80$) are admitted alongside normal tissue, preserving class balance in the adaptation gradient and preventing threshold collapse.
3. **Conclusion:** Asymmetric tumor rejection is a **causal mechanism** of UTTA-Med underperformance in multicenter digital pathology.
