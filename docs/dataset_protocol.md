# Dataset Protocol — UTTA-Med

**Single source of truth for data handling. Never violate these rules.**

## Primary Dataset
Camelyon17-WILDS (official `wilds` package).

## Critical Rule — Hospital / Split Mapping
- **Never hard-code hospital numbers** from any project document or memory.
- Load with the official WILDS loader:
  ```python
  from wilds import get_dataset
  dataset = get_dataset(dataset="camelyon17", download=True)
  ```
- Inspect actual split arrays, domain IDs, and metadata.
- Record the verified mapping in `configs/camelyon17.yaml`.
- That YAML file is the only authoritative mapping for the entire project.
- Document the verified mapping in the paper’s Dataset section with a short table.

## Experimental Data Separation (Frozen)
```
Source hospitals          → TRAINING
OOD validation hospital   → τ selection, LR, adaptation steps, MC passes, batch size, early stopping
Target hospital (unlabeled) → TTA adaptation only
Target labels             → FINAL EVALUATION ONLY
```

## Absolute Prohibitions
Target labels must **never** be used for:
- training
- adaptation
- threshold (τ) selection
- hyperparameter selection
- model selection
- any form of early stopping or checkpoint selection on the target domain

## Leakage Rules
- Train WSI ∩ Validation WSI = ∅
- Train WSI ∩ Test WSI = ∅
- Validation WSI ∩ Test WSI = ∅
- Never re-split patches across hospitals or WSIs.
- Use only the official WILDS splits.

## Secondary Datasets (Development Only)
- PathMNIST — initial pipeline debugging
- PCam — rapid prototyping, MC-Dropout & Grad-CAM testing
- MedMNIST-C — optional robustness check (after main Camelyon17 experiments)

PCam and PathMNIST are **not** substitutes for the primary cross-hospital claim.
