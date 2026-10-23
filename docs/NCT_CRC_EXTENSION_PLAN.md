# Phase 4 (optional) — NCT-CRC multi-class extension

**Purpose:** Stretch venue (MICCAI / IEEE JBHI). Show that on **multi-class** histopathology, entropy / MC-Dropout can diverge from max-softmax confidence.  
**Do not** mix NCT numbers into Camelyon Table 1. New table only.  
**Skip entirely** if targeting ISBI / EMBC / MIDL only.

---

## 1. Dataset

| Item | Spec |
|---|---|
| Name | NCT-CRC-HE-100K (and/or CRC-VAL-HE-7K for held-out) |
| Task | 9-class tissue classification |
| Images | 224×224 H&E tiles (resize to 224; ImageNet norm) |
| Split (proposed) | Official or stratified 70/15/15 by class; **no hospital IDs** — use a **corruption or site proxy** if available, else standard ID split + synthetic shift (stain/brightness) for TTA stress |
| Honest framing | Multi-class **in-domain + mild shift** study of *gates*, not Camelyon-style hospital WILDS |

If you need a true domain shift: train on NCT-CRC-HE-100K, test on CRC-VAL-HE-7K (different staining/source), which is a common setup.

---

## 2. Frozen protocol (mirror Camelyon)

| Item | Value |
|---|---|
| Backbone | ResNet-18, ImageNet init |
| Head | 9-logit linear (not 1-logit) |
| Loss | Cross-entropy |
| Dropout | 0.5 before head (same as Camelyon) |
| Seeds | At least `{42, 123, 2024}` (5 if GPU allows) |
| TTA | BN affine only, **running stats frozen**, Adam `lr=1e-5`, `k=1` |
| Gates | Random / confidence (max softmax) / entropy / MC-Dropout variance (N=20) at **matched val coverage** |
| τ / thresholds | Frozen on **validation** only; never on test |
| EATA | Optional baseline with published-style filtering |

**Do not** retune Camelyon τ from NCT.

---

## 3. What to measure (new table)

| Metric | Why |
|---|---|
| Accuracy / macro-F1 / ECE | Multi-class standard |
| Harm analog | Fraction of batches where adapted Acc < source Acc − margin (define margin e.g. 1%) |
| Coverage | % patches adapted |
| Gate agreement | Spearman rank correlation: confidence vs entropy vs U_MC |
| Matched-coverage comparison | Same as Camelyon RQ6 |

**Success for the extension (not required to “beat” confidence):**

- Report whether U_MC **separates** from confidence on 9-class (rank correlation ≪ 1).  
- Report whether selective gates still reduce harm vs ungated Tent.  
- If Tent is stable on NCT, say so — do not force a collapse narrative.

---

## 4. Minimal code work

```text
src/data/nct_crc.py          # Dataset + splits
configs/nct_crc.yaml         # paths, num_classes=9
scripts/train_source_nct.py  # or extend existing trainer for CE / 9-class
scripts/run_tta_nct.py       # Tent + gates, matched coverage
```

Reuse:

- `src/tta/tent.py`, `gates.py`, `uncertainty_gated.py` (generalize head from 1-logit sigmoid to multi-class softmax where needed)
- BN freeze + affine-only update path already in Camelyon code

**Watch:** binary-only helpers (e.g. 1−|2p−1|) must not be used as the only multi-class U; use entropy and MC variance properly.

---

## 5. GPU budget (order of magnitude)

| Step | Rough time (single 16–24 GB GPU) |
|---|---|
| Source train × 3 seeds | ~few hours total |
| TTA eval all gates × 3 seeds | ~1–3 hours |
| Total | ~1–2 days including debug |

---

## 6. Paper placement (when you write)

- Main paper Camelyon tables **unchanged**.  
- New subsection or appendix: “Multi-class extension (NCT-CRC)”.  
- Claim: *protocol transfers; gate rankings may differ when |Y|>2*.  
- Do **not** claim hospital-level WILDS on NCT.

---

## 7. Decision gate

| If you want… | Do NCT-CRC? |
|---|---|
| ISBI / EMBC / MIDL soon | **No** — ship Camelyon + Phase-1 |
| MICCAI / JBHI | **Yes** — complete §2–5 first |
| Foundation-model TTA | **No** — that is a different paper |

---

## 8. Immediate next step if you start NCT

1. Download NCT-CRC-HE-100K (+ CRC-VAL-HE-7K if using as target).  
2. Add `configs/nct_crc.yaml` with `num_classes: 9`.  
3. Train one seed source; verify ID accuracy is sane (>0.9 typical).  
4. Only then wire TTA + gates.

Ask when you have the data path and we can draft `nct_crc.yaml` + dataset stub against your tree.
