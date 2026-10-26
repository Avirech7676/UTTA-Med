import json
from collections import defaultdict

def classify_file(r, dupes):
    p = r['path']
    sz = r['size']
    h = r['sha256']
    
    # ==========================================
    # C. DELETE CANDIDATE (Requires approval)
    # ==========================================
    # 1. Pycache & Pytest cache
    if '__pycache__' in p or p.endswith('.pyc') or '.pytest_cache' in p:
        return 'DELETE_CANDIDATE', 'TEMPORARY_CACHE', 'Python bytecode cache and pytest cache. Regenerated automatically at runtime.'
    
    # 2. Exact Duplicates with Canonical Version Retained
    if p == 'paper/ieee/utta_med_h5_manuscript_draft.pdf':
        return 'DELETE_CANDIDATE', 'EXACT_DUPLICATE_PDF', 'Exact byte-for-byte duplicate of paper/ieee/utta_med_camera_ready.pdf. Safe to delete.'
    if p == 'results/kaggle_training/checkpoints/utta_med_v2_resnet18_adapted_s42.pt':
        return 'DELETE_CANDIDATE', 'EXACT_DUPLICATE_CHECKPOINT', 'Exact duplicate (43.84 MB) of canonical checkpoints/utta_med_v2_resnet18_adapted_s42.pt.'
    if p.startswith('tmp/'):
        return 'DELETE_CANDIDATE', 'TEMPORARY_SCRATCH', 'Temporary Kaggle kernel push folders and redundant logs/evidence duplicates.'
    
    # 3. Duplicate Overlays across Seeds
    # For MIDOG overlays, seed_42 and seed_123 are identical copies of seed_7
    if 'overlays/' in p and ('seed_42' in p or 'seed_123' in p):
        if h in dupes and len(dupes[h]) > 1:
            canonical = sorted(dupes[h])[0]
            if canonical != p:
                return 'DELETE_CANDIDATE', 'EXACT_DUPLICATE_OVERLAY', f'Redundant identical copy of canonical {canonical}. Consumes 51.34 MB.'

    # ==========================================
    # B. ARCHIVE (Historical / Provenance Value)
    # ==========================================
    if p == 'UTTA_MED_V2_REPRODUCIBILITY_PACKAGE.zip':
        return 'ARCHIVE', 'LARGE_ARCHIVE_BUNDLE', 'Full zip backup (47.50 MB) of earlier v2 results package.'
    if p.startswith('results/archive/'):
        return 'ARCHIVE', 'OLD_RESULT', 'Superseded v1 legacy CSVs and statistics already placed in archive.'
    if p.startswith('results/deprecated/'):
        return 'ARCHIVE', 'OLD_DOC', 'Deprecated result notes from earlier development phases.'
    if p in ('paper/UTTA_MED_PAPER_DRAFT.pdf', 'paper/ieee/utta_med.tex', 'paper/ieee/utta_med.bib', 'paper/ieee/utta_med.bbl', 'paper/utta_med.bib'):
        return 'ARCHIVE', 'OLD_PAPER_DRAFT', 'Superseded initial manuscript draft and v1 bibliography files.'
    if p.startswith('results/upgrade_p0/'):
        return 'ARCHIVE', 'OLD_DIAGNOSTIC', 'Interim P0 diagnostic run tables replaced by final v2 tables.'
    if p.startswith('figures/') or p.startswith('paper/figures/') or p.startswith('paper/figs/'):
        return 'ARCHIVE', 'OLD_FIGURE', 'Legacy/intermediate figure assets. Canonical figures are in paper/ieee/figs/ and results/final/figures/.'
    if p.startswith('results/figures/') and not p.startswith('results/final/figures/'):
        return 'ARCHIVE', 'OLD_FIGURE', 'Legacy experiment plots from earlier runs.'

    # ==========================================
    # A. KEEP (Active & Submission Critical)
    # ==========================================
    if p.startswith('paper/ieee/'):
        return 'KEEP', 'ACTIVE_MANUSCRIPT', 'Active submission LaTeX source, figures, tables, and camera-ready PDF.'
    if p.startswith('src/'):
        return 'KEEP', 'CORE_CODE', 'Production codebase, adaptation engines, and gating policies.'
    if p.startswith('tests/'):
        return 'KEEP', 'TEST_SUITE', 'Full pytest suite (150 test cases, 100% passing).'
    if p.startswith('configs/'):
        return 'KEEP', 'EXPERIMENT_CONFIG', 'Active YAML configurations for all benchmarked pipelines.'
    if p.startswith('checkpoints/'):
        return 'KEEP', 'FINAL_CHECKPOINTS', 'Core trained source and adapted weights supporting paper claims.'
    if p.startswith('results/final/'):
        return 'KEEP', 'ACTIVE_FINAL_RESULTS', 'Authoritative final figures, tables, and manuscript assets.'
    if p.startswith('results/v2/'):
        return 'KEEP', 'ACTIVE_FINAL_RESULTS', 'Official v2 benchmark stages verified by verify_manuscript_tables.py.'
    if p.startswith('results/tables/') or p.startswith('results/statistical/'):
        return 'KEEP', 'STATISTICAL_PROVENANCE', 'Underpinning statistical distributions, p-values, and table manifests.'
    if p.startswith('results/data/'):
        return 'KEEP', 'PROCESSED_DATA', 'WSI bootstrap NPZ arrays needed to regenerate confidence intervals.'
    if p.startswith('results/metrics/') or p.startswith('results/ablations/') or p.startswith('results/predictions/') or p.startswith('results/training/'):
        return 'KEEP', 'EXPERIMENTAL_RESULTS', 'Run metrics, ablation records, and prediction arrays supporting paper claims.'
    if p.startswith('results/optional/'):
        return 'KEEP', 'ABLATION_SUPPORT', 'Corruption robustness and secondary benchmark metrics.'
    if p.startswith('results/kaggle_training/'):
        return 'KEEP', 'GPU_AUDIT_TRAIL', 'Audit evidence of real NVIDIA T4 GPU Kaggle runs.'
    if p.startswith('results/generalization/'):
        return 'KEEP', 'GENERALIZATION_BENCHMARKS', 'Cross-scanner MIDOG benchmark results and canonical visual overlays.'
    if p.startswith('data/'):
        return 'KEEP', 'BENCHMARK_ANNOTATIONS', 'Dataset split definitions and spatial centroid ground truths.'
    if p.startswith('scripts/'):
        return 'KEEP', 'AUTOMATION_SCRIPTS', 'Verification, figure generation, and compilation scripts.'
    if p.startswith('notebooks/'):
        return 'KEEP', 'REPRODUCIBILITY_NOTEBOOKS', 'Kaggle execution scripts and training pipelines.'
    if p.startswith('docs/') or p.startswith('reproducibility/'):
        return 'KEEP', 'DOCUMENTATION_AND_PROTOCOLS', 'Technical specifications, protocols, and audit dossiers.'
    if p.startswith('experiments/') or p.startswith('logs/'):
        return 'KEEP', 'EXPERIMENT_LOGS', 'Seed execution records, commit hashes, and training histories.'
    if p.startswith('.github/'):
        return 'KEEP', 'CI_CONFIG', 'Continuous integration workflow configuration.'
    if '/' not in p:
        return 'KEEP', 'ROOT_METADATA', 'Core repository governance: README, LICENSE, environment, metadata.'
    if p.startswith('paper/'):
        return 'KEEP', 'MANUSCRIPT_METADATA', 'Submission checklist, metadata dossiers, publication tables.'

    # ==========================================
    # D. NEEDS MANUAL REVIEW
    # ==========================================
    return 'NEEDS_MANUAL_REVIEW', 'UNCLASSIFIED', 'Unclassified file requiring manual audit.'

def main():
    with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = data['files']
    dupes = data['duplicates']

    classified = defaultdict(list)
    cat_sizes = defaultdict(int)

    for r in files:
        cat, subcat, reason = classify_file(r, dupes)
        classified[cat].append((r, subcat, reason))
        cat_sizes[cat] += r['size']

    total_workspace = sum(r['size'] for r in files)
    print("=== FINAL CLASSIFICATION SUMMARY ===")
    for cat in ['KEEP', 'ARCHIVE', 'DELETE_CANDIDATE', 'NEEDS_MANUAL_REVIEW']:
        items = classified[cat]
        sz_mb = cat_sizes[cat] / (1024 * 1024)
        pct = (cat_sizes[cat] / total_workspace) * 100
        print(f"{cat:<20}: {len(items):>5} files | {sz_mb:>8.2f} MB ({pct:>5.1f}%)")

    # Output detailed classification JSON
    output_data = {}
    for cat in classified:
        output_data[cat] = [
            {
                'path': r['path'],
                'size': r['size'],
                'mtime': r['mtime'],
                'sha256': r['sha256'],
                'subcat': subcat,
                'reason': reason
            }
            for r, subcat, reason in classified[cat]
        ]
    with open('results/workspace_classification_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print("Saved results/workspace_classification_detailed.json")

if __name__ == '__main__':
    main()
