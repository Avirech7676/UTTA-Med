import os
import hashlib
import json
import time
from collections import defaultdict

def main():
    with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = data['files']
    dupes = data['duplicates']
    file_by_path = {r['path']: r for r in files}

    # 1. Root files
    print("=== ROOT FILES ===")
    for r in files:
        if '/' not in r['path']:
            sz_mb = r['size'] / (1024 * 1024)
            print(f"{r['path']:<45} | {sz_mb:>8.2f} MB | {r['mtime']}")

    # 2. Checkpoints
    print("\n=== CHECKPOINTS ===")
    for r in files:
        if r['path'].startswith('checkpoints/'):
            sz_mb = r['size'] / (1024 * 1024)
            sha = r['sha256'][:16] if r['sha256'] not in ('ERROR', 'LARGE_NOT_HASHED') else r['sha256']
            print(f"{r['path']:<55} | {sz_mb:>8.2f} MB | {sha}...")

    # 3. Paper directory
    print("\n=== PAPER DIRECTORY BREAKDOWN ===")
    paper_subdirs = defaultdict(int)
    paper_sub_sizes = defaultdict(int)
    for r in files:
        if r['path'].startswith('paper/'):
            parts = r['path'].split('/')
            sub = parts[1] if len(parts) > 2 else '[root_paper]'
            paper_subdirs[sub] += 1
            paper_sub_sizes[sub] += r['size']
    for sub, count in sorted(paper_subdirs.items()):
        print(f"paper/{sub:<20} | {count:>4} files | {paper_sub_sizes[sub]/(1024*1024):>6.2f} MB")

    # 4. Results directory breakdown
    print("\n=== RESULTS DIRECTORY BREAKDOWN ===")
    results_subdirs = defaultdict(int)
    results_sub_sizes = defaultdict(int)
    for r in files:
        if r['path'].startswith('results/'):
            parts = r['path'].split('/')
            sub = parts[1] if len(parts) > 2 else '[root_results]'
            results_subdirs[sub] += 1
            results_sub_sizes[sub] += r['size']
    for sub, count in sorted(results_subdirs.items()):
        print(f"results/{sub:<20} | {count:>4} files | {results_sub_sizes[sub]/(1024*1024):>6.2f} MB")

    # 5. Notebooks and Docs
    print("\n=== NOTEBOOKS ===")
    for r in files:
        if r['path'].startswith('notebooks/'):
            print(f"{r['path']:<55} | {r['size']/1024:>8.1f} KB | {r['mtime']}")

    # 6. Large/Important Duplicates
    print("\n=== DUPLICATE GROUPS (Key Files) ===")
    for h, paths in dupes.items():
        first_r = file_by_path[paths[0]]
        sz = first_r['size']
        if sz > 20000 or any('checkpoint' in p or 'table' in p or 'fig' in p or 'final' in p or p.endswith('.pdf') for p in paths):
            print(f"Hash {h[:12]} ({sz/1024:.1f} KB, {len(paths)} copies):")
            for p in paths:
                print(f"   - {p}")

    # 7. Tmp, cache, pycache files
    pycache_files = [r for r in files if '__pycache__' in r['path'] or r['path'].endswith('.pyc')]
    pytest_files = [r for r in files if '.pytest_cache' in r['path']]
    tmp_files = [r for r in files if r['path'].startswith('tmp/')]
    print(f"\n=== CACHE & TEMPORARY FILES ===")
    print(f"Pycache files: {len(pycache_files)} files, {sum(r['size'] for r in pycache_files)/(1024*1024):.2f} MB")
    print(f"Pytest cache:  {len(pytest_files)} files, {sum(r['size'] for r in pytest_files)/(1024*1024):.2f} MB")
    print(f"tmp/ folder:   {len(tmp_files)} files, {sum(r['size'] for r in tmp_files)/(1024*1024):.2f} MB")

if __name__ == '__main__':
    main()
