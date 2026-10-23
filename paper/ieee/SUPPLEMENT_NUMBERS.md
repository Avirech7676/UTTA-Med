# Supplement numbers (locked)

## Camelyon17-C (seed 42, 4000+/4000−)

Identity AUROC 0.937 matches full hospital-2 source 0.935.

TTA does not restore severe C. Under gaussian_noise-3:
- Tent harm 0.081, F1 0.463
- UTTA harm 0.057, F1 0.519, coverage ~0.30
- UTTA ≈ confidence

Blur-3: AUROC 0.928 but F1 0.464 (recall 0.303). Do not call this blur-robust from AUROC.

## PathMNIST-C (9-class, no TTA)

Clean acc 0.920 / AUROC-OVR 0.992.
Severity 3: noise 0.145, brightness 0.120, contrast 0.135, blur 0.784, jpeg 0.773.

Not hospital shift. Not Camelyon17. Appendix paragraph only.
