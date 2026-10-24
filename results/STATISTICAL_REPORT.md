# Five-seed statistical close-out (locked protocol)

No retuning. Seeds `{42, 123, 2024, 7, 99}`. Hospital 2 only.
BN stats frozen, TTA lr \(10^{-5}\), \(N_\text{MC}=20\), \(\tau=\) per-seed 70th percentile of val \(U\).

Primary multi-seed inference: **paired seed-level \(\Delta\)** (df = 4, \(t_{0.975}=2.776\)).
Primary WSI inference: **paired slide-resample of \(\Delta\)** on seed-42 dumps (`y,p,slide`); other seeds use per-slide paired bootstrap (not pooled patches).

---

## 1–2. Merged mean ± SD (Table 1)

| Method | AUROC | F1 | ECE | Coverage | Harm |
|---|---:|---:|---:|---:|---:|
| Source-only | 0.930 ± 0.009 | 0.807 ± 0.025 | 0.128 ± 0.026 | — | — |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.284 ± 0.179 | 1.000 ± 0.000 | 0.187 ± 0.179 |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.152 ± 0.023 | 0.885 ± 0.034 | 0.025 ± 0.015 |
| Random-gated | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.298 ± 0.166 | 0.700 ± 0.001 | 0.194 ± 0.170 |
| Confidence-gated | **0.936 ± 0.022** | **0.831 ± 0.032** | 0.130 ± 0.030 | 0.581 ± 0.047 | **0.007 ± 0.003** |
| **UTTA-Med** | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.131 ± 0.030 | 0.575 ± 0.047 | **0.007 ± 0.003** |

Entropy ≡ confidence (binary). Shift: ID→target AUROC drop **0.070 ± 0.009**. \(U\)–error \(r=0.397\pm0.016\).

Tent/Random SD is large because **seeds 7 and 99 collapsed**. That is data, not a bug. Keep them.

---

## 3. Per-seed \(\Delta\) UTTA − Source (do not retune)

| Seed | \(\Delta\) AUROC | \(\Delta\) F1 | harm UTTA | harm Tent | cov UTTA | source sens |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | **+0.0152** | **+0.0347** | 0.0032 | 0.059 | 0.562 | 0.750 |
| 123 | **+0.0072** | **+0.0272** | 0.0063 | 0.044 | 0.583 | 0.734 |
| 2024 | **+0.0235** | **+0.0283** | 0.0070 | 0.068 | 0.581 | 0.680 |
| **7** | **−0.0118** | **+0.0162** | 0.0097 | **0.375** | 0.510 | **0.638** |
| **99** | **−0.0051** | **+0.0092** | 0.0103 | **0.391** | 0.641 | 0.681 |

AUROC: **3/5 up, 2/5 down**. F1: **5/5 up**. Harm: **5/5 ≪ Tent**.

---

## 4–5. Paired CIs

### Seed-level (n = 5, primary for the paper table)

| Contrast | Mean \(\Delta\) | 95% t CI | t p | Wilcoxon p | Cohen’s d | signs |
|---|---:|---|---:|---:|---:|---|
| UTTA − Source **AUROC** | +0.0058 | [−0.012, +0.024] | 0.42 | 0.44 | 0.40 | 3+/2− |
| UTTA − Confidence **AUROC** | −0.0000 | [−0.0008, +0.0007] | 0.87 | 0.63 | −0.08 | 1+/4− |
| UTTA − EATA AUROC | +0.0145 | [+0.002, +0.027] | 0.033 | 0.062 | 1.43 | 5+/0 |
| UTTA − Tent AUROC | +0.237 | [−0.141, +0.616] | 0.16 | 0.062 | 0.78 | 5+/0 |
| UTTA − Source **F1** | **+0.0231** | **[+0.010, +0.036]** | **0.007** | 0.062 | **2.26** | **5+/0** |
| UTTA − Confidence F1 | −0.0010 | [−0.0015, −0.0005] | 0.005 | 0.062 | −2.45 | 0+/5− |
| UTTA − Tent **harm** | −0.180 | [−0.399, +0.039] | 0.085 | 0.062 | −1.02 | 0+/5− |

Wilcoxon two-sided with n = 5 cannot go below **0.0625** when all signs agree. That is a sample-size floor, not a failed F1 result.

Holm (8 primary tests): UTTA vs Source F1 **0.051**; UTTA vs Confidence F1 **0.043** (effect size **0.001** — significant and negligible); all AUROC contrasts **n.s.**

### Seed-42 paired **WSI** \(\Delta\) (pooled patches, 10 slides, 1000 boots)

| Contrast | \(\Delta\) AUROC [95%] | \(\Delta\) F1 [95%] |
|---|---|---|
| UTTA − Source | **+0.014 [+0.002, +0.026]** | **+0.040 [+0.018, +0.090]** |
| Confidence − Source | **+0.015 [+0.002, +0.026]** | **+0.040 [+0.018, +0.083]** |
| UTTA − Confidence | **−0.0007 [−0.0012, −0.0004]** | −0.001 [−0.002, −0.001] |
| UTTA − Tent | +0.027 [−0.049, +0.136] | — |
| Tent − Source | −0.016 [−0.128, +0.063] | — |

On seed 42, gated TTA vs source **excludes 0** at the slide level. UTTA vs confidence excludes 0 but the gap is **< 0.001**.

Per-slide paired bootstrap on 123 / 2024 / 7 / 99 (equal-weight slides): UTTA−Source CI **excludes 0 only on 2024**; **includes 0 on 123, 7, 99**. Same story as the 3+/2− seed split.

---

## 6. UTTA vs confidence / entropy (required)

They are the same operating point on this binary task.

- AUROC mean \(\Delta = 0.000\), CI width 0.0015
- F1: confidence wins by **0.001** (detectable, not meaningful)
- Harm: 0.007 vs 0.007
- Coverage: 0.575 vs 0.581
- Entropy ≡ confidence on every seed

**RQ6 answer (locked):** MC-Dropout is a valid error signal (\(r\approx0.40\)) but **does not beat** a confidence gate at matched coverage. Report as a **tie**, not a loss of the paper.

---

## 7–8. Harm and coverage (aggregated)

| | Harm mean ± SD | Coverage |
|---|---:|---:|
| Tent | 0.187 ± 0.179 | 1.00 |
| Random | 0.194 ± 0.170 | 0.70 |
| EATA | 0.025 ± 0.015 | 0.89 |
| Confidence | **0.007 ± 0.003** | 0.581 ± 0.047 |
| UTTA-Med | **0.007 ± 0.003** | 0.575 ± 0.047 |

Harm reduction vs Tent is **~25×** on the mean and **5/5 seeds**. The t-interval on \(\Delta\) harm still includes 0 because Tent’s variance is dominated by two collapses. That is why harm should be reported as **mean ± SD + 5/5 sign count**, not as a single p-value.

Random at 70% coverage (and at 56% on seed 42) does **not** get this harm profile → not “adapt on fewer samples”.

---

## 9. Why seeds 7 and 99 drop AUROC (no retune)

Both still **raise F1** and **block Tent collapse**. AUROC is not universal.

**Seed 7** is the weakest source: target AUROC 0.913, sensitivity **0.638**, F1 0.773 (vs ~0.83 on other seeds). OOD val was still 0.978 — the source **overfit the val AUROC / under-recalled tumors on hospital 2**.

Tent **collapsed on 9/10 slides** (worst: slide 29, \(\Delta\)AUROC **−0.73**). Random collapsed with it.

UTTA on seed 7:

- Slide **28** (31,878 patches, 85% tumor): source 0.940 → UTTA **0.955** (helps the dominant slide)
- Slide **29** (12,742 patches, 52% tumor): source 0.738 → UTTA **0.626** (this slide drives the pooled AUROC drop)
- Slide 24 (rare tumor): source 0.566 → UTTA **0.602** (helps a hard low-prevalence slide)
- Equal-weight mean \(\Delta\)AUROC across slides: **−0.007** (small); pooled drop is **slide 29**

**Seed 99** is the same geometry: Tent ~0 on slide 29; UTTA −0.092 there; modest gains on 23/24/27/28. Source sensitivity 0.681.

Mechanism (consistent with Grad-CAM): gating **stops entropy-min from zeroing logits** (no collapse), but on a weak/high-shift source the accepted high-confidence patches still **raise specificity / cut recall**. F1 can go up while AUROC dips. That is a real, reportable limitation — not a reason to drop the seeds.

---

## 10. How to write Table 1

Use mean ± SD. Add one sentence:

> WSI-level 95% CIs (n = 10 hospital-2 slides) are wide and overlap source vs gated TTA on AUROC except on seed 42, where the paired \(\Delta\) excludes 0. Tent collapse on seeds 7 and 99 is slide-significant. The reproducible endpoints are **F1 (5/5)** and **harm (~25× vs Tent)**, not universal AUROC.

Do **not** use the old seed-level t-interval that gave Tent AUROC CI up to 1.10.

---

## 11. Ablations / Grad-CAM (already locked)

| Item | Result |
|---|---|
| MC-N | N=20 frozen (corr 0.425 vs 0.437 at N=50) |
| Random @ UTTA coverage 0.562 | AUROC 0.922, harm 0.057 vs UTTA 0.951 / 0.003 |
| Inner steps k | k=1 official; k≥5 collapses Tent **and** UTTA |
| Source Grad-CAM | audit; high-U errors near p=0.5 |
| Frozen-source UTTA CAM | AUROC 0.950; 0 harm events in CAM set |
| UMAP | supporting only |

No new GPU. Next is the manuscript, not another seed.
