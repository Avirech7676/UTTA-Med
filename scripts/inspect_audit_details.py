import json
import os
from collections import defaultdict

with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

files = data['files']
dupes = data['duplicates']

print("=== CHECKING PAPER SUBDIRECTORIES ===")
for r in files:
    if r['path'].startswith('paper/') and ('pdf' in r['path'] or 'tex' in r['path'] or 'md' in r['path']):
        print("%-60s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['mtime']))

print("\n=== CHECKING NOTEBOOKS ===")
for r in files:
    if r['path'].startswith('notebooks/'):
        print("%-60s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['mtime']))

print("\n=== CHECKING RESULTS/KAGGLE_TRAINING ===")
for r in files:
    if r['path'].startswith('results/kaggle_training/'):
        print("%-60s | %8.2f MB | %s" % (r['path'], r['size']/(1024*1024), r['sha256'][:16]))

print("\n=== CHECKING TMP DIRECTORY ===")
for r in files:
    if r['path'].startswith('tmp/'):
        print("%-60s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['sha256'][:16]))

print("\n=== CHECKING DOCS DIRECTORY (SAMPLE) ===")
docs_files = [r for r in files if r['path'].startswith('docs/')]
print(f"Total docs files: {len(docs_files)}")
for r in docs_files[:15]:
    print("%-60s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['mtime']))
