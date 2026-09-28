# Leakage Rules — UTTA-Med

**Non-negotiable. Violation invalidates the paper.**

1. **Official WILDS splits only**  
   Never create custom train/val/test splits of Camelyon17 patches.

2. **WSI-level separation**  
   ```
   Train WSI ∩ Validation WSI = ∅
   Train WSI ∩ Test WSI = ∅
   Validation WSI ∩ Test WSI = ∅
   ```
   Enforced by `tests/test_no_leakage.py`.

3. **Hospital / domain isolation**  
   Source hospitals used only for training.  
   OOD validation hospital used only for hyperparameter / τ selection.  
   Target hospital unlabeled data used only for adaptation.  
   Target labels used only for final evaluation.

4. **No target-label leakage into any decision**  
   - No τ selection on target labels  
   - No early stopping on target labels  
   - No model selection on target labels  
   - No temperature scaling fitted on target labels  
   - No coverage matching that peeks at target labels

5. **No slide-level leakage across any split**  
   Patches from the same WSI never appear in more than one of {train, val, test}.

6. **Reproducibility**  
   Every reported number must be traceable to a logged experiment ID, Git commit, config, and checkpoint.
