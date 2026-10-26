import json
import os
import re

with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

files = data['files']

src_files = [r for r in files if r['path'].startswith('src/') and r['path'].endswith('.py')]
scripts_files = [r for r in files if r['path'].startswith('scripts/') and r['path'].endswith('.py')]
test_files = [r for r in files if r['path'].startswith('tests/') and r['path'].endswith('.py')]

print(f"Total src .py files: {len(src_files)}")
print(f"Total scripts .py files: {len(scripts_files)}")
print(f"Total tests .py files: {len(test_files)}")

# Check for hardcoded old paths in scripts
old_patterns = [
    'table_main_results_final.csv',
    'table_per_seed.csv',
    'UTTA-MED',
    'table1_main_results_source.csv',
    'confidence_intervals.csv',
    'v1_old_results',
    'results/final'
]

print("\n=== SCANNING SCRIPTS FOR LEGACY/ARCHIVE/FINAL REFERENCES ===")
for r in scripts_files:
    fp = r['path']
    try:
        with open(fp, 'r', encoding='utf-8') as fh:
            content = fh.read()
        matched = []
        for pat in old_patterns:
            if pat in content:
                matched.append(pat)
        if matched:
            print(f"{fp:<50} | Matches: {', '.join(matched)}")
    except Exception as e:
        pass
