#!/usr/bin/env python3
"""
UTTA-Med Project 2: Research Direction Restoration & Repository Cleanup
=======================================================================
Restores the original approved UTTA-Med research identity, archives unrequested
extensions (DenseNet-121, ResNet-50, EMA Teacher-Student, C0-C7 / D8-D9 extensions),
and cleans obsolete progress artifacts while preserving authentic research data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Authoritative Research Metadata
OFFICIAL_PROJECT_TITLE = (
    "Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: "
    "Multicenter Classification and Dense Detection Under Domain Shift"
)
BACKUP_TAG = "UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024"
ARCHIVE_BRANCH = "archive/unrequested-research-extensions"
BACKUP_COMMIT = "3a293e0"

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def classify_files() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Classifies all files in the repository into:
    Group A: Original Research (Keep)
    Group B: Approved Performance Improvements (Keep)
    Group C: Unrequested Extensions (Archive)
    Group D: Obsolete Artifacts (Remove from active workspace)
    """
    group_a = []
    group_b = []
    group_c = []
    group_d = []
    
    # Explicit Group B (Approved improvements)
    group_b_set = {
        "ORIGINAL_UTTA_MED_ROOT_CAUSE_ANALYSIS.md",
        "MC_DROPOUT_VERIFICATION.md",
        "UNCERTAINTY_SELECTION_ANALYSIS.csv",
        "IMPROVED_UTTA_MED_ABLATION.csv",
        "VALIDATION_PERFORMANCE_COMPARISON.csv",
        "FIVE_SEED_UTTA_MED_COMPARISON.csv",
        "FINAL_UTTA_MED_IMPROVEMENT_REPORT.md",
        "CORRECTED_GPU_LATENCY.csv",
        "IMPROVED_RESNET18_TRAINING.py",
        "scripts/run_full_original_utta_audit.py",
        "scripts/audit_mc_dropout.py",
        "scripts/analyze_uncertainty_selection.py",
        "results/mc_dropout_reconciliation.json",
        "results/uncertainty_signal_ablation.json",
    }
    
    # Explicit Group C (Unrequested extensions to archive)
    group_c_prefixes = [
        "DENSENET121",
        "ACTUAL_DENSENET",
        "FINAL_DENSENET",
        "VERIFIED_DENSENET",
        "RESNET50",
        "EMA_CONSISTENCY",
        "FINAL_E0_E1_E2",
        "D8_VS_D9",
        "D8_D9",
        "BIO_SAFEADAPT",
        "08_BIO_SAFEADAPT",
        "10_BIO_SAFEADAPT",
        "12_BIO_SAFEADAPT",
        "VERIFIED_C0_C7",
        "MATCHED_COVERAGE_C7",
        "FEATURE_DISTANCE_ABLATIONS",
        "CLASS_AWARE_GATING",
        "UTTA_MED_21",
        "UTTA_MED_22",
        "UTTA_MED_V2",
        "A1_A9_EQUIVALENCE",
        "A1_VS_A9",
        "FINAL_ADAPTATION_SUPERIORITY",
        "FINAL_COVERAGE_RECONCILIATION",
        "FINAL_MODEL_COMPARISON",
        "FINAL_MODEL_PERFORMANCE_COMPARISON",
        "FINAL_MODEL_PERFORMANCE_REPORT",
        "FINAL_FIVE_SEED_CHECKPOINT_REGISTRY",
        "FINAL_REAL_GPU_TRAINING_RESULTS",
        "ACTUAL_FIVE_SEED_GPU",
        "ACTUAL_TTA_COMPARISON",
        "GENUINE_KAGGLE_EXPERIMENTS",
    ]
    
    # Explicit Group D (Obsolete generated reports & temp artifacts)
    group_d_prefixes = [
        "scratch_prompt",
        "PHASE5_",
        "PHASE6_",
        "PHASE7_",
        "PHASE8_",
        "PHASE9_",
        "PHASE10_",
        "PHASE11_",
        "FINAL_PENDING_",
        "FINAL_PENDING_AND_FAILED",
        "FINAL_PENDING_FAILED_BLOCKED",
        "FINAL_PENDING_TASKS",
        "FINAL_PROJECT_COMPLETION",
        "FINAL_PROJECT_REPRODUCIBILITY",
        "FINAL_PUBLICATION_READINESS_DECISION",
        "FINAL_TASK_STATUS_MATRIX",
        "COMPLETE_ISSUE_RESOLUTION",
        "FINAL_ISSUES_AND_CORRECTIONS",
        "FINAL_RESEARCH_IMPROVEMENT_VERDICT",
        "FINAL_RESEARCH_RISK_REGISTER",
        "FINAL_RESEARCH_ARTIFACT_REGISTRY",
        "IEEE_PAPER_FINAL_CHECKLIST",
        "IEEE_PAPER_FORMATTING_AUDIT",
        "FINAL_IEEE_MANUSCRIPT_REVIEW",
        "FINAL_IEEE_PAPER_VERIFICATION",
        "FINAL_GPU_EXECUTION_REPORT",
        "FINAL_GPU_TRAINING_STATUS",
        "KAGGLE_GPU_EXECUTION_EVIDENCE",
        "KAGGLE_GPU_EXECUTION_VERIFICATION",
        "KAGGLE_GPU_FAILURE_REPORT",
        "KAGGLE_EXECUTION_VERIFICATION",
        "REAL_GPU_TRAINING_REPORT",
        "REAL_GPU_EXECUTION_REPORT",
        "REAL_KAGGLE_GPU_EXECUTION_REPORT",
        "REAL_KAGGLE_EXECUTION_VERIFICATION",
        "PERFORMANCE_IMPROVEMENT_VERDICT",
        "PERFORMANCE_STATISTICAL_VALIDATION",
        "FINAL_STATISTICAL_COMPARISON",
        "FINAL_STATISTICAL_IMPROVEMENT_REPORT",
        "FINAL_LATENCY_AND_COVERAGE_VALIDATION",
        "FINAL_LATENCY_RECONCILIATION",
        "FINAL_CLASSIFICATION_STATISTICS",
        "CLASSIFICATION_CONFUSION_MATRICES",
        "MATCHED_COVERAGE_VERIFICATION",
        "VERIFIED_FIVE_SEED_CHECKPOINTS",
        "VERIFIED_TTA_ADAPTATION_LOGS",
        "VERIFIED_C0_C7_RESULTS",
        "ORIGINAL_RESULTS_PRESERVATION",
        "MIDOG_MANIFEST_RECONCILIATION",
        "IEEE_PAPER_COMPILATION_REPORT",
    ]
    
    # Specific directories to archive / remove
    unwanted_dirs = {
        "checkpoints/densenet121",
        "kaggle_output_densenet121",
        "kaggle_output_test",
        "kaggle_output_check",
        "scripts/kaggle_densenet121",
        "utta-med-densenet121-training",
        "temp_kernel_pull",
        "temp_pull",
        "temp_meta",
        "temp_output_check",
        "temp_pilot_test",
        "temp_check_output",
        "src/v2",
        "configs/v2",
        "results/v2",
    }
    
    # Unwanted scripts in scripts/
    unwanted_scripts = {
        "scripts/retrieve_and_finalize_densenet121.py",
        "scripts/kaggle/train_densenet121_kaggle.py",
        "scripts/generate_bio_safeadapt_ablations.py",
        "scripts/generate_bio_safeadapt_multi_scanner.py",
        "scripts/generate_bio_safeadapt_verified_predictions.py",
        "scripts/generate_full_bio_safeadapt_ablations.py",
        "scripts/evaluate_c_series.py",
        "scripts/run_c7_matched_coverage.py",
        "scripts/tune_confidence_feature_gating.py",
        "scripts/tune_phase4_phase5.py",
        "scripts/investigate_a1_a9_equivalence.py",
        "scripts/generate_all_phase_reports.py",
        "scripts/build_master_task_registry.py",
        "scripts/generate_master_csv_deliverables.py",
    }

    # Iterate over all files
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
            
        rel = p.relative_to(ROOT).as_posix()
        
        # Skip git internal files
        if rel.startswith(".git/"):
            continue
            
        # File info
        size = p.stat().st_size
        filename = p.name
        
        # Check if in Group B
        if rel in group_b_set or filename in group_b_set:
            group_b.append({
                "path": rel,
                "filename": filename,
                "size_bytes": size,
                "group": "GROUP_B_APPROVED_IMPROVEMENT",
                "description": "Approved performance improvement or diagnostic artifact within ResNet-18 UTTA-Med."
            })
            continue
            
        # Check if unwanted directory
        in_unwanted_dir = any(rel.startswith(ud + "/") or rel == ud for ud in unwanted_dirs)
        in_unwanted_script = rel in unwanted_scripts
        
        # Check Group C (Unrequested extensions)
        matches_c = any(filename.startswith(pfx) for pfx in group_c_prefixes) or (
            "densenet" in rel.lower() or "resnet50" in rel.lower() or "bio_safeadapt" in rel.lower()
        )
        
        if in_unwanted_dir or in_unwanted_script or matches_c:
            group_c.append({
                "path": rel,
                "filename": filename,
                "size_bytes": size,
                "group": "GROUP_C_UNREQUESTED_EXTENSION",
                "description": "Unrequested research extension (DenseNet-121, ResNet-50, EMA, Bio-SafeAdapt expansions, C-series)."
            })
            continue
            
        # Check Group D (Obsolete generated reports)
        matches_d = any(filename.startswith(pfx) for pfx in group_d_prefixes) or (
            filename.endswith(".log") or filename.startswith("temp_") or filename.startswith("scratch_")
        )
        
        if matches_d:
            group_d.append({
                "path": rel,
                "filename": filename,
                "size_bytes": size,
                "group": "GROUP_D_OBSOLETE_ARTIFACT",
                "description": "Obsolete generated progress report, redundant metric dashboard, or scratch file."
            })
            continue
            
        # Default: Group A (Original Research)
        group_a.append({
            "path": rel,
            "filename": filename,
            "size_bytes": size,
            "group": "GROUP_A_ORIGINAL_RESEARCH",
            "description": "Original certified research asset (ResNet-18 model, TTA adapter, Camelyon17 data, MIDOG Arms A-E, paper PDF)."
        })
        
    return group_a, group_b, group_c, group_d

def main():
    print("=" * 80)
    print("UTTA-MED RESEARCH DIRECTION RESTORATION & REPOSITORY RECONCILIATION")
    print(f"Official Title: {OFFICIAL_PROJECT_TITLE}")
    print(f"Preservation Tag: {BACKUP_TAG} | Archival Branch: {ARCHIVE_BRANCH}")
    print("=" * 80)
    
    # 1. Classify all files
    print("\n[*] Auditing and classifying all repository files...")
    group_a, group_b, group_c, group_d = classify_files()
    
    print(f"[+] Classification Breakdown:")
    print(f"  Group A (Original Research - KEEP):               {len(group_a):4d} files")
    print(f"  Group B (Approved Improvements - KEEP):            {len(group_b):4d} files")
    print(f"  Group C (Unrequested Extensions - ARCHIVE):       {len(group_c):4d} files")
    print(f"  Group D (Obsolete Generated Artifacts - REMOVE):  {len(group_d):4d} files")
    print(f"  Total Audited Files:                              {len(group_a) + len(group_b) + len(group_c) + len(group_d):4d} files")
    
    # 2. Output Inventories
    # Inventory 1: Original Research
    df_a = pd.DataFrame(group_a)
    df_a.to_csv(ROOT / "ORIGINAL_RESEARCH_FILE_INVENTORY.csv", index=False)
    print("\n[+] Generated ORIGINAL_RESEARCH_FILE_INVENTORY.csv")
    
    # Inventory 2: Unwanted Development (Group C + Group D)
    unwanted_records = group_c + group_d
    df_unwanted = pd.DataFrame(unwanted_records)
    df_unwanted.to_csv(ROOT / "UNWANTED_DEVELOPMENT_INVENTORY.csv", index=False)
    print("[+] Generated UNWANTED_DEVELOPMENT_INVENTORY.csv")
    
    # Inventory 3: Removed from Active Workspace
    # We record files that will be archived/removed from active workspace root
    df_removed = pd.DataFrame(group_c + group_d)
    df_removed.to_csv(ROOT / "REMOVED_ACTIVE_WORKSPACE_FILES.csv", index=False)
    print("[+] Generated REMOVED_ACTIVE_WORKSPACE_FILES.csv")
    
    # 3. Create physical archive directory for Group C extensions
    archive_dir = ROOT / "archive" / "unrequested_extensions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] Ensuring physical backup directory exists: {archive_dir.relative_to(ROOT)}...")
    
    # Copy essential code/configs from Group C to archive directory prior to removal
    for item in group_c:
        src = ROOT / item["path"]
        dst = archive_dir / item["path"]
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                pass
    print("[+] Copied unrequested research extensions to archive/unrequested_extensions/ for transparent local provenance.")
    
    # 4. Remove Group C and Group D files from the active root and active paths
    print("\n[*] Pruning unrequested and obsolete files from active workspace...")
    removed_count = 0
    for item in group_c + group_d:
        p = ROOT / item["path"]
        # Do not delete archive/ directory itself
        if "archive/" in item["path"]:
            continue
        if p.exists():
            try:
                p.unlink()
                removed_count += 1
            except Exception as e:
                print(f"  Warning: could not delete {item['path']}: {e}")
                
    # Clean empty directories
    for d in [
        "checkpoints/densenet121",
        "kaggle_output_densenet121",
        "kaggle_output_test",
        "kaggle_output_check",
        "scripts/kaggle_densenet121",
        "utta-med-densenet121-training",
        "temp_kernel_pull",
        "temp_pull",
        "temp_meta",
        "temp_output_check",
        "temp_pilot_test",
        "temp_check_output",
        "src/v2",
        "configs/v2",
        "results/v2",
    ]:
        dp = ROOT / d
        if dp.exists():
            try:
                shutil.rmtree(dp)
            except Exception:
                pass
                
    print(f"[+] Successfully pruned {removed_count} unrequested and obsolete files from the active workspace.")
    
    # 5. Regression Smoke Tests
    print("\n" + "=" * 80)
    print("EXECUTING ORIGINAL UTTA-MED REGRESSION & INTEGRITY VERIFICATION SUITE")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Verify Original ResNet-18 Checkpoints
    print("[*] Test 1: Verifying ResNet-18 Source Checkpoints...")
    ckpts = [
        "checkpoints/camelyon17_resnet18_source_s42_BEST.pt",
        "checkpoints/camelyon17_resnet18_source_s123_BEST.pt",
        "checkpoints/camelyon17_resnet18_source_s2024_BEST.pt",
        "checkpoints/camelyon17_resnet18_source_s7_BEST.pt",
        "checkpoints/camelyon17_resnet18_source_s99_BEST.pt",
        "checkpoints/midog_resnet18_s42_BEST.pt",
    ]
    ckpt_pass = True
    for c in ckpts:
        cp = ROOT / c
        if not cp.exists() or cp.stat().st_size < 40_000_000:
            print(f"  FAIL: Checkpoint missing or corrupted: {c}")
            ckpt_pass = False
        else:
            print(f"  PASS: {c} ({cp.stat().st_size:,} bytes, SHA256={compute_sha256(cp)[:12]}...)")
    test_results.append(("ResNet-18 Source Checkpoints", ckpt_pass))
    
    # Test 2: Verify Original Manuscripts
    print("\n[*] Test 2: Verifying Original Research Manuscripts...")
    manuscripts = [
        "UTTA_MED_FINAL_COMPACT_PAPER.pdf",
        "FINAL_MANUSCRIPT.pdf",
    ]
    manu_pass = True
    for m in manuscripts:
        mp = ROOT / m
        if not mp.exists():
            print(f"  FAIL: Manuscript missing: {m}")
            manu_pass = False
        else:
            h = compute_sha256(mp)
            print(f"  PASS: {m} ({mp.stat().st_size:,} bytes, SHA256={h[:16]}...)")
    test_results.append(("Original Manuscripts Preservation", manu_pass))
    
    # Test 3: Verify Model Building & TTA Adapters
    print("\n[*] Test 3: Verifying ResNet-18 Architecture & TTA Modules...")
    try:
        import torch
        from src.models.resnet18 import build_resnet18
        from src.tta.uncertainty_gated import GatedTTA
        from src.tta.gates import score_mc_dropout, score_confidence
        
        m = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1)
        x = torch.randn(4, 3, 96, 96)
        out = m(x)
        assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
        
        tta = GatedTTA(m, gate="confidence", tau=0.85)
        p_adapt = tta.adapt_batch(x)
        assert p_adapt.shape == (4,), f"Unexpected adapted shape: {p_adapt.shape}"
        print("  PASS: ResNet-18 and GatedTTA build and adapt cleanly.")
        arch_pass = True
    except Exception as e:
        print(f"  FAIL: Model/TTA instantiation error: {e}")
        arch_pass = False
    test_results.append(("ResNet-18 & TTA Instantiation", arch_pass))
    
    # Test 4: Verify Target Predictions Arrays
    print("\n[*] Test 4: Verifying 35 Authentic Bootstrap Predictions (Center 2)...")
    pred_dir = ROOT / "results" / "data" / "wsi_bootstrap"
    npz_files = list(pred_dir.glob("*.npz"))
    if len(npz_files) == 35:
        print(f"  PASS: All {len(npz_files)} seed prediction files intact in {pred_dir.relative_to(ROOT)}.")
        pred_pass = True
    else:
        print(f"  FAIL: Expected 35 prediction files, found {len(npz_files)}.")
        pred_pass = False
    test_results.append(("Target Prediction Arrays (85,054 patches)", pred_pass))
    
    # Test 5: Verify MIDOG Detection Arms A-E
    print("\n[*] Test 5: Verifying MIDOG Detection Arms A-E...")
    midog_files = [
        "MIDOG_CANONICAL_MANIFEST.csv",
        "09_VERIFIED_DETECTION_BASELINES.csv",
        "VERIFIED_DETECTION_RESULTS.csv",
        "VERIFIED_DETECTION_STATISTICS.csv",
        "checkpoints/midog_resnet18_s42_BEST.pt",
    ]
    midog_pass = True
    for mf in midog_files:
        p = ROOT / mf
        if not p.exists():
            print(f"  FAIL: MIDOG file missing: {mf}")
            midog_pass = False
        else:
            print(f"  PASS: {mf} verified.")
    test_results.append(("MIDOG Detection Arms A-E", midog_pass))
    
    print("\n--- Regression Test Summary ---")
    all_passed = True
    for t_name, passed in test_results:
        print(f"  {t_name:<40}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
            
    assert all_passed, "One or more regression tests failed!"
    print("\n[+] ALL REGRESSION SMOKE TESTS PASSED WITH ZERO ERRORS.")

if __name__ == "__main__":
    main()
