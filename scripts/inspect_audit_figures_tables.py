import json
from collections import defaultdict

with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

files = data['files']
dupes = data['duplicates']

print("=== FIGURE DIRECTORIES COMPARISON ===")
fig_dirs = ['figures', 'paper/figures', 'paper/figs', 'paper/ieee/figs', 'results/figures', 'results/final/figures']
for fd in fig_dirs:
    flist = [r for r in files if r['path'].startswith(fd + '/') and r['path'].count('/') == fd.count('/') + 1]
    sz = sum(r['size'] for r in flist) / (1024 * 1024)
    print("%-30s | %3d files | %6.2f MB" % (fd, len(flist), sz))

print("\n=== PAPER ROOT DIRECTORY FILES ===")
paper_root_files = [r for r in files if r['path'].startswith('paper/') and r['path'].count('/') == 1]
for r in paper_root_files:
    print("%-50s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['mtime']))

print("\n=== CONFIGS DIRECTORY FILES ===")
for r in files:
    if r['path'].startswith('configs/'):
        print("%-50s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['mtime']))

print("\n=== RESULTS/TABLES VS RESULTS/FINAL/TABLES ===")
tables_v1 = [r for r in files if r['path'].startswith('results/tables/')]
tables_final = [r for r in files if r['path'].startswith('results/final/tables/')]
print(f"results/tables/: {len(tables_v1)} files")
print(f"results/final/tables/: {len(tables_final)} files")
