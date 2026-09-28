# Uncertainty-Gated Test-Time Adaptation for Cross-Hospital Histopathology Classification

**UTTA-Med** — working manuscript (numbers frozen from 5-seed Camelyon17-WILDS protocol)

> Do not invent metrics. All test numbers below are from seeds {42, 123, 2024, 7, 99}.
> Target hospital labels were used only for final evaluation.

---

## Abstract

Deep networks trained on histopathology patches from one set of hospitals degrade on an unseen hospital. Test-time adaptation (TTA) can recover some of that drop, but entropy minimization on every target sample can also amplify errors. We study whether *restricting* TTA to low-uncertainty samples makes adaptation safer under real Camelyon17-WILDS shift (source hospitals {0,3,4} → target hospital 2).

We train ResNet-18 source models (5 seeds), then compare source-only, Tent, EATA, and four gates (random, softmax confidence, predictive entropy, MC-Dropout variance) under a frozen protocol: BatchNorm affine updates only, running stats frozen, TTA learning rate $10^{-5}$, and $\tau$ chosen as the 70th percentile of hospital-1 uncertainty. Gate thresholds for random/confidence/entropy are matched on hospital 1 only.

On hospital 2, source AUROC is $0.930\pm0.009$. Fair Tent is unstable: it **collapses on 2/5 seeds** (mean AUROC $0.698\pm0.324$, harm rate $0.187\pm0.179$). Selective gates prevent collapse. Confidence-gated TTA and UTTA-Med (MC-Dropout) both reach AUROC $0.936/0.935$, raise F1 on **all five seeds**, and cut harm to $0.007\pm0.003$. UTTA-Med does **not** outperform softmax confidence ($\Delta$ AUROC $<0.001$ every seed). Random gating also collapses on 2/5 seeds, so the benefit is *which* samples are used, not merely using fewer of them.

**Takeaway.** Under this binary Camelyon17 protocol, *gating* is what makes TTA reliable; MC-Dropout variance is a valid error signal but is redundant with confidence. The scientific contribution is the leakage-controlled reliability profile (coverage, harm, calibration, matched-coverage controls), not a claim that Bayesian uncertainty uniquely wins.

---

## 1. Introduction

A model trained on labeled patches from source hospitals is often deployed on a new scanner, stain, and patient population. Formally $P_{\mathrm{source}}(X,Y)\neq P_{\mathrm{target}}(X,Y)$. Test-time adaptation (TTA) updates a frozen source checkpoint using unlabeled target images, typically by entropy minimization on BatchNorm affine parameters (Tent). The known failure mode is **harmful adaptation**: a confident wrong prediction becomes an update signal and the model degrades.

We ask whether predictive uncertainty can be used as a *hard gate*: adapt only if $U(x)<\tau$, otherwise skip. The method (UTTA-Med) estimates $U(x)$ by MC-Dropout predictive variance. We do not claim this is the first uncertainty-aware TTA method. Closest prior work includes EATA (entropy + diversity filtering) and SAR (sharpness-aware TTA). Our difference is: (i) an explicit MC-Dropout variance gate; (ii) matched-coverage controls against random, confidence, and entropy gates; (iii) joint reporting of coverage, harm rate, and calibration on a leakage-controlled Camelyon17-WILDS split.

**Contributions**

1. A frozen, leakage-free Camelyon17-WILDS TTA protocol (official hospitals; target labels unused until the final test pass).
2. Evidence that **ungated Tent is unstable** even with conservative BN-affine updates (collapse on 2/5 seeds).
3. Evidence that **selective TTA** (confidence or MC-Dropout) prevents collapse, cuts harm ~25× vs Tent, and improves F1 on 5/5 seeds.
4. A negative-but-informative gate-control result: **MC-Dropout ≈ softmax confidence** for this binary ResNet-18.

---

## 2. Related work (positioning)

**Tent** minimizes prediction entropy at test time, updating BatchNorm affine parameters. **EATA** filters high-entropy samples and reweights by diversity. **SAR** additionally uses sharpness-aware optimization. Several medical TTA papers use entropy or confidence filters; we do not claim primacy.

UTTA-Med uses stochastic predictive *variance* (MC-Dropout) as a hard participation gate and evaluates the gate against random / confidence / entropy at matched validation coverage. EATA is kept as a strong published-style baseline under the *same* BN-affine update scope so that filtering strategy is not confounded with which parameters move.

---

## 3. Method

### 3.1 Data and leakage

Camelyon17-WILDS 96×96 RGB patches, binary tumor / non-tumor. Official mapping (verified from the HuggingFace parquet dump, not from memory):

| Role | Hospitals | n patches | n WSIs |
|---|---|---:|---:|
| Source train | 0, 3, 4 | 302,436 | 30 |
| In-domain val | 0, 3, 4 | 33,560 | 30 (same slides, held-out patches) |
| OOD val (tune only) | **1** | 34,904 | 10 |
| Target test | **2** | 85,054 | 10 |

Train ∩ OOD-val slides = ∅, train ∩ test = ∅, OOD-val ∩ test = ∅.

Hospital 1 is used for early stopping, $\tau$, and matched-coverage thresholds. Hospital 2 labels are touched **once**, after those choices are frozen.

### 3.2 Source model

ResNet-18 (ImageNet init), 512-D GAP, head `Linear–ReLU–Dropout(0.5)–Linear` with **1 logit**, BCE-with-logits. Adam $10^{-4}$, batch 128, up to 20 epochs, patience 5 on hospital-1 AUROC, **save-best-only**. Seeds {42, 123, 2024, 7, 99}.

### 3.3 TTA protocol (all methods)

Continual one-pass over the target stream (`shuffle=False`), one SGD/Adam step per batch. **Only BN $\gamma,\beta$** are trainable; BN running mean/var are **frozen** (`bn_freeze_stats`). TTA lr $=10^{-5}$ selected on hospital 1 after a larger-lr grid collapsed. Empty-gate batches skip `backward()` and `step()`.

### 3.4 UTTA-Med gate

Keep dropout active. $N=20$ stochastic forward passes:

$$
\bar p(x)=\frac1N\sum_{j=1}^N p_j,\qquad U(x)=\frac1N\sum_{j=1}^N (p_j-\bar p)^2.
$$

$$
w(x)=\mathbf{1}[U(x)<\tau],\qquad
L=\frac{\sum_i w_i\,H(p_i)}{\sum_i w_i}\quad(\text{skip if }\sum w_i=0).
$$

$\tau$ = 70th percentile of $U$ on hospital 1 (rule frozen after seed 42; re-estimated per seed, not copied as an absolute threshold).

### 3.5 Controls

- **Tent:** $w\equiv 1$.
- **EATA:** entropy margin $e=0.4$ (BN-affine only).
- **Random:** adapt each sample with probability = UTTA coverage on hospital 1 (~0.70).
- **Confidence:** adapt if $\max(p,1-p) \ge t$, $t$ set to matched coverage on hospital 1.
- **Entropy:** adapt if $H(p)\le t_H$, matched the same way.

For a Bernoulli probability, $H(p)$ is a strictly monotone function of $\max(p,1-p)$, so entropy and confidence gates are expected to coincide. They did, on every seed.

---

## 4. Results

### 4.1 RQ1 — Distribution shift

In-domain AUROC is $\approx 0.999$. Target (hospital 2) source-only AUROC is **$0.930\pm0.009$** (drop $0.070\pm0.009$). F1 drop is larger ($0.18\pm0.02$): the source model becomes **under-sensitive** on the unseen hospital (typical target recall $0.64$–$0.75$ vs specificity $>0.95$).

### 4.2 RQ3 — Uncertainty vs error (hospital 1, before gating)

MC-Dropout $U$ increases monotonically with error on all 5 seeds (Pearson $r=0.397\pm0.016$). Lowest-uncertainty decile error $\approx 0\%$; highest-decile error $35$–$42\%$. The uncertainty signal is informative; gating is justified.

### 4.3 RQ2 / RQ4 — TTA and harmful adaptation

**Table 1.** Hospital 2, mean $\pm$ std over 5 seeds.

| Method | AUROC | F1 | ECE | Harm | Cov. |
|---|---:|---:|---:|---:|---:|
| Source-only | $0.930\pm0.009$ | $0.807\pm0.025$ | $0.128\pm0.026$ | — | — |
| Tent | $0.698\pm0.324$ | $0.506\pm0.411$ | $0.284\pm0.179$ | $0.187\pm0.179$ | 1.00 |
| EATA | $0.921\pm0.026$ | $0.809\pm0.031$ | $0.152\pm0.024$ | $0.025\pm0.015$ | 0.89 |
| Random gate | $0.697\pm0.304$ | $0.487\pm0.385$ | $0.298\pm0.166$ | $0.194\pm0.170$ | 0.70 |
| Confidence gate | $\mathbf{0.936\pm0.022}$ | $\mathbf{0.831\pm0.032}$ | $0.130\pm0.030$ | $\mathbf{0.007\pm0.003}$ | 0.58 |
| UTTA-Med | $0.935\pm0.022$ | $0.830\pm0.032$ | $0.131\pm0.031$ | $0.007\pm0.003$ | 0.58 |

Tent with the *conservative* protocol still **collapsed on seeds 7 and 99** (AUROC $0.34$, F1 $0.05$, harm $\approx 0.38$). On the other three seeds Tent is roughly source-level. Mean Tent AUROC is therefore not a point estimate of a stable method; it is a mixture of “fine” and “destroyed”. EATA never collapsed but did not beat source AUROC.

Selective gates **never collapsed**. Harm fell from $0.187$ (Tent) to $0.007$ (confidence / UTTA). Random gating at 70% coverage collapsed on the same two seeds as Tent: reducing the number of updates is not sufficient.

### 4.4 RQ5 — Calibration and F1

Gated TTA **improves F1 vs source on 5/5 seeds** (mean $+0.024$). AUROC improves on 3/5 seeds and is slightly below source on the two collapse-prone seeds. ECE is not a win vs source ($0.130$ vs $0.128$); it *is* a win vs collapsing Tent. Reliability is therefore primarily about **harm and F1**, not ECE.

### 4.5 RQ6 — Is MC-Dropout better than confidence?

No. $|\mathrm{AUROC}_{\mathrm{UTTA}}-\mathrm{AUROC}_{\mathrm{conf}}|<0.001$ on every seed. Entropy ≡ confidence, as predicted for binary outputs. The ablation still matters: it shows that a cheap confidence gate captures essentially all of the benefit of MC-Dropout variance for this backbone and task. UTTA-Med remains a valid instantiation of “uncertainty-gated TTA”; it is not a new accuracy champion.

### 4.6 RQ7 — Grad-CAM audit (seed-42 source, hospital 2)

Grad-CAM on the last residual stage, 512 shuffled hospital-2 patches (215 TP, 226 TN, 15 FP, 56 FN). This is an **interpretability audit, not clinical validation**. 96×96 maps are coarse.

**What the galleries show**

- **Low-uncertainty correct tumors** ($p=1$, $U\sim10^{-15}$): CAM sits on dense nuclei, not fat/white space.
- **Low-uncertainty correct normals**: fat, stroma, sparse cells; CAM is often blank. That is expected for a 1-logit head when the logit is strongly negative (near-zero gradient).
- **True positives**: CAM on cellular regions; some maps collapse to a center blob (resolution limit).
- **False positives**: lymphocyte-dense / reactive cellularity that looks tumor-like at 96×96. Several have $p>0.96$ — **confident errors that would pass a confidence/UTTA gate**.
- **False negatives**: missed tumor. Two kinds: (i) high-$U$, $p\approx0.3$–$0.5$ (gate would skip); (ii) low-$U$, $p\approx0.01$ (confident miss — gate would **accept**). Gating cannot fix confidently wrong samples.
- **High-$U$ errors**: mixed FP/FN with $p$ near the decision boundary and CAMs on edges/corners. These are the samples the gate is designed to exclude from the TTA loss.

**Paper caption (Figure 7).** Source ResNet-18, seed 42, hospital 2. Columns: image, Grad-CAM, overlay. Rows illustrate TP / TN / FP / FN, plus lowest-$U$ correct and highest-$U$ incorrect. Blank maps on confident negatives are a 1-logit Grad-CAM artifact, not evidence that the model “sees nothing.” Audit only.

---

---

## 5. Discussion

The positive scientific statement is: **under real hospital shift, unrestricted entropy-minimization TTA is unsafe, and restricting updates to high-confidence / low-variance samples makes it safe and improves F1.**

The honest qualifier is: **the particular Bayesian estimator (MC-Dropout variance) did not beat a softmax-confidence gate.** For a binary, well-trained ResNet, predictive variance and $|p-0.5|$ rank samples similarly (the U–error correlation is real, but so is confidence–error). That is useful to report. A reviewer who knows EATA will look for this comparison; it is in the table.

**Limitations.** (i) One dataset, one backbone, binary labels. (ii) Test CIs are seed-level; primary WSI-level bootstrap needs per-patch dumps (10 test slides). (iii) Grad-CAM is a 96×96 audit on the source model only (no post-TTA pair) and often blanks on confident negatives. (iv) Confidence/entropy/UTTA coverage on test (~0.51–0.65) is below the val-matched 0.70 because hospital 2 is more uncertain — report both. (v) Tent lr was selected on hospital 1 after larger lrs collapsed; we do not claim this is Tent’s original ImageNet hyperparameter. (vi) Confident false negatives ($p\approx0$) can still pass the gate.

---

## 6. Conclusion

Uncertainty-aware *selection* makes TTA reliable on Camelyon17-WILDS; unrestricted Tent does not. MC-Dropout is a legitimate gate, but not a better one than confidence here. The result is still worth publishing if it is written that way.

---

## Reproducibility

- Splits: official WILDS / HF `wltjr1007/Camelyon17-WILDS` parquet, filtered by `center`.
- Seeds: 42, 123, 2024, 7, 99.
- Checkpoints: `camelyon17_resnet18_source_s{seed}_BEST.pt` (save-best on hospital 1).
- Protocol: `bn_freeze_stats`, TTA lr $1\mathrm{e}{-5}$, $N_{\mathrm{MC}}=20$, $\tau=$ val 70th percentile of $U$.
- Every headline number traces to `camelyon17_resnet18_full_s{seed}.json`.
