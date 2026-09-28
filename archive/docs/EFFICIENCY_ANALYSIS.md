# Computational efficiency (rechecked)

ResNet-18 BatchNorm affine tensors were previously listed as **106**. That count is **ResNet-50**. Measured from `collect_bn_affine`:

| Backbone | BN modules (approx.) | Affine tensors (γ,β) | Affine parameter scalars |
|---|---:|---:|---:|
| **ResNet-18 (official)** | 20 | **40** | **9,600** |
| ResNet-50 (optional) | 53 | **106** | **53,120** |

Paper wording: *“TTA updates 40 BN affine tensors (9.6k scalars) on ResNet-18.”*

MC-Dropout N=20 is ~20× extra forwards for the **gate only**. Confidence gating is the cheap equivalent (RQ6 tie).

Do not quote 18.4 ms / 384 ms as hardware-universal; those were one T4 run. Relative 20× is the stable claim.
