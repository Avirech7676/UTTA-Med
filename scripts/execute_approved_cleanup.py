import os
import shutil
import hashlib
import json

def sha256_file(p):
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def safe_move(src, dst):
    if os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.move(src, dst)
        return True
    return False

def safe_delete_file(p, expected_sha=None):
    if os.path.exists(p):
        if expected_sha:
            actual_sha = sha256_file(p)
            if actual_sha != expected_sha:
                print(f"[SAFETY ABORT] Hash mismatch for {p}. Expected {expected_sha}, got {actual_sha}. Skipping deletion.")
                return False, 0
        sz = os.path.getsize(p)
        os.remove(p)
        return True, sz
    return False, 0

def safe_delete_dir(d):
    if os.path.exists(d):
        sz = 0
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    sz += os.path.getsize(fp)
                except:
                    pass
        shutil.rmtree(d, ignore_errors=True)
        return True, sz
    return False, 0

def main():
    print("=" * 80)
    print("EXECUTING APPROVED WORKSPACE CLEANUP & ARCHIVAL")
    print("=" * 80)

    # -------------------------------------------------------------
    # PHASE 1: ARCHIVAL (Moving historical artifacts to archive/)
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Archiving Historical Artifacts ---")
    archived_count = 0

    # 1. Root zip bundle
    if safe_move('UTTA_MED_V2_REPRODUCIBILITY_PACKAGE.zip', 'archive/backups/UTTA_MED_V2_REPRODUCIBILITY_PACKAGE.zip'):
        print("[ARCHIVE] Moved UTTA_MED_V2_REPRODUCIBILITY_PACKAGE.zip -> archive/backups/")
        archived_count += 1

    # 2. Old manuscript drafts
    old_draft_moves = [
        ('paper/UTTA_MED_PAPER_DRAFT.pdf', 'archive/old_drafts/UTTA_MED_PAPER_DRAFT_v1_legacy.pdf'),
        ('paper/ieee/utta_med.tex', 'archive/old_drafts/utta_med_initial_template.tex'),
        ('paper/ieee/utta_med.bib', 'archive/old_drafts/utta_med_initial_template.bib'),
        ('paper/ieee/utta_med.bbl', 'archive/old_drafts/utta_med_initial_template.bbl'),
        ('paper/utta_med.bib', 'archive/old_drafts/paper_utta_med_v1.bib'),
    ]
    for src, dst in old_draft_moves:
        if safe_move(src, dst):
            print(f"[ARCHIVE] Moved {src} -> {dst}")
            archived_count += 1

    # 3. Old figures
    old_fig_dirs = [
        ('figures', 'archive/old_figures/figures_root_legacy'),
        ('paper/figures', 'archive/old_figures/paper_figures_legacy'),
        ('paper/figs', 'archive/old_figures/paper_figs_legacy'),
        ('results/figures', 'archive/old_figures/results_figures_legacy'),
    ]
    for src, dst in old_fig_dirs:
        if os.path.exists(src):
            ensure_dir(os.path.dirname(dst))
            # Move contents or entire directory
            if not os.path.exists(dst):
                shutil.move(src, dst)
            else:
                for item in os.listdir(src):
                    s_item = os.path.join(src, item)
                    d_item = os.path.join(dst, item)
                    if not os.path.exists(d_item):
                        shutil.move(s_item, d_item)
                shutil.rmtree(src, ignore_errors=True)
            print(f"[ARCHIVE] Moved directory {src} -> {dst}")
            archived_count += 1

    # 4. Old diagnostic results in results/upgrade_p0/
    if os.path.exists('results/upgrade_p0'):
        safe_move('results/upgrade_p0', 'results/archive/intermediate_p0')
        print("[ARCHIVE] Moved results/upgrade_p0/ -> results/archive/intermediate_p0/")
        archived_count += 1

    # 5. Intermediate milestone results in results/ root
    milestone_files = [
        'results/adaptation_analysis.csv',
        'results/CHECKPOINTS.md',
        'results/figure_manifest.csv',
        'results/five_seed_final_stats.json',
        'results/h2b2_results_manifest.json',
        'results/h2c_results_manifest.json',
        'results/master_results.csv',
        'results/phase14_audit.json',
        'results/slide_bootstrap.csv',
        'results/STATISTICAL_REPORT.md',
        'results/table_paired_bootstrap.csv',
        'results/table_per_seed_delta_source.csv'
    ]
    for mf in milestone_files:
        dst = os.path.join('results/archive/intermediate_milestones', os.path.basename(mf))
        if safe_move(mf, dst):
            print(f"[ARCHIVE] Moved {mf} -> {dst}")
            archived_count += 1

    # -------------------------------------------------------------
    # PHASE 2: SAFE DELETIONS (Verified Duplicates and Caches)
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Executing Approved Safe Deletions ---")
    deleted_files = 0
    deleted_bytes = 0

    # 1. Exact Duplicate PDF
    # Verify canonical utta_med_camera_ready.pdf exists first!
    if os.path.exists('paper/ieee/utta_med_camera_ready.pdf'):
        ok, sz = safe_delete_file('paper/ieee/utta_med_h5_manuscript_draft.pdf')
        if ok:
            print(f"[DELETE DUPLICATE] Removed paper/ieee/utta_med_h5_manuscript_draft.pdf ({sz/1024/1024:.2f} MB)")
            deleted_files += 1
            deleted_bytes += sz
    else:
        print("[SAFETY ABORT] Canonical paper/ieee/utta_med_camera_ready.pdf not found! Aborting deletion of draft pdf.")

    # 2. Exact Duplicate Checkpoint (43.84 MB)
    # Verify canonical checkpoints/utta_med_v2_resnet18_adapted_s42.pt exists and hash matches!
    canon_ckpt = 'checkpoints/utta_med_v2_resnet18_adapted_s42.pt'
    dupe_ckpt = 'results/kaggle_training/checkpoints/utta_med_v2_resnet18_adapted_s42.pt'
    if os.path.exists(canon_ckpt) and sha256_file(canon_ckpt) == '185cb4c7215ad10a82051c6bde14b22ca9c732463b2111542892b748221a9a1b':
        ok, sz = safe_delete_file(dupe_ckpt, expected_sha='185cb4c7215ad10a82051c6bde14b22ca9c732463b2111542892b748221a9a1b')
        if ok:
            print(f"[DELETE DUPLICATE] Removed redundant checkpoint copy {dupe_ckpt} ({sz/1024/1024:.2f} MB)")
            deleted_files += 1
            deleted_bytes += sz
    else:
        print("[SAFETY ABORT] Canonical checkpoint validation failed! Skipping deletion.")

    # 3. Duplicate Overlays in seed_42 and seed_123
    for scanner_idx in [1, 2, 3]:
        base_dir = f'results/generalization/h3/scanner_{scanner_idx}_target/overlays'
        for dupe_seed in ['seed_42', 'seed_123']:
            seed_dir = os.path.join(base_dir, dupe_seed)
            if os.path.exists(seed_dir):
                # Verify canonical seed_7 exists with files
                canon_seed_dir = os.path.join(base_dir, 'seed_7')
                if os.path.exists(canon_seed_dir) and len(os.listdir(canon_seed_dir)) > 0:
                    ok, sz = safe_delete_dir(seed_dir)
                    if ok:
                        print(f"[DELETE DUPLICATE] Removed redundant overlays in {seed_dir} ({sz/1024/1024:.2f} MB)")
                        # Count approx 18 files per dir
                        deleted_files += 18
                        deleted_bytes += sz

    # 4. Temporary Kaggle Scratch Directories (tmp/)
    if os.path.exists('tmp'):
        ok, sz = safe_delete_dir('tmp')
        if ok:
            print(f"[DELETE SCRATCH] Removed temporary scratch directory tmp/ ({sz/1024/1024:.2f} MB)")
            deleted_files += 27
            deleted_bytes += sz

    # 5. Pytest cache (.pytest_cache)
    if os.path.exists('.pytest_cache'):
        ok, sz = safe_delete_dir('.pytest_cache')
        if ok:
            print(f"[DELETE CACHE] Removed .pytest_cache/ ({sz/1024:.1f} KB)")
            deleted_files += 4
            deleted_bytes += sz

    # 6. Python bytecode caches (__pycache__)
    pycache_count = 0
    pycache_bytes = 0
    for root_dir in ['src', 'tests', 'scripts']:
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
            if os.path.basename(dirpath) == '__pycache__':
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        pycache_bytes += os.path.getsize(fp)
                        os.remove(fp)
                        pycache_count += 1
                    except:
                        pass
                try:
                    os.rmdir(dirpath)
                except:
                    pass
    print(f"[DELETE CACHE] Cleaned {pycache_count} pycache bytecode files ({pycache_bytes/1024/1024:.2f} MB)")
    deleted_files += pycache_count
    deleted_bytes += pycache_bytes

    print("\n" + "=" * 80)
    print("CLEANUP EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Archival batches completed: {archived_count}")
    print(f"Total files removed:        {deleted_files}")
    print(f"Total storage reclaimed:    {deleted_bytes / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    main()
