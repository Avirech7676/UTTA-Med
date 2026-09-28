# Architecture freeze (MUST FIX — resolved)

There is **no ResNet-17**. Official backbone is **ResNet-18**. ResNet-50 is optional Phase 11.

| Knob | Official | Illegal leftovers |
|---|---|---|
| Output | **1 logit**, `BCEWithLogitsLoss`, Sigmoid at eval | `num_classes: 2` softmax |
| Dropout | **0.5** in the head only | `dropout_p: 0.2` |
| Source optimizer | **Adam** lr=`1e-4`, wd=`1e-4` | AdamW |
| TTA optimizer | **Adam** lr=`1e-5` | Adam lr=`1e-3` (collapses) |
| BN affine tensors (R18) | **40** (20 BN × γ,β) | “106 tensors” (that is ResNet-50) |
| τ | 70th percentile of **hospital-1 U**, per seed | global `tau: 0.05` |

All five `*_BEST.pt` files store `classifier.3.weight` with shape **`[1, 512]`**.

Canonical files: `configs/resnet18.yaml`, `configs/authoritative_resnet18_utta.yaml`, `configs/tau_by_seed.yaml`.

`configs/source_resnet18.yaml` is marked **obsolete** (it used to contain 2-class / dropout 0.2 / AdamW).
