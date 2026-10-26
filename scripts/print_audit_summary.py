import json
from collections import defaultdict

with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

files = data['files']

print('=== 1. ROOT FILES ===')
for r in files:
    if '/' not in r['path']:
        sz = r['size'] / (1024 * 1024)
        print('%-45s | %8.2f MB | %s' % (r['path'], sz, r['mtime']))

print('\n=== 2. CHECKPOINTS ===')
for r in files:
    if r['path'].startswith('checkpoints/'):
        sz = r['size'] / (1024 * 1024)
        h = r['sha256'][:16] if r['sha256'] not in ('ERROR', 'LARGE_NOT_HASHED') else r['sha256']
        print('%-55s | %8.2f MB | %s' % (r['path'], sz, h))

print('\n=== 3. RESULTS SUBDIRECTORIES ===')
res_dirs = defaultdict(lambda: [0, 0])
for r in files:
    if r['path'].startswith('results/'):
        parts = r['path'].split('/')
        sub = parts[1] if len(parts) > 2 else '[root]'
        res_dirs[sub][0] += 1
        res_dirs[sub][1] += r['size']

for sub in sorted(res_dirs.keys()):
    c, sz = res_dirs[sub]
    print('results/%-25s | %4d files | %8.2f MB' % (sub, c, sz / (1024 * 1024)))

print('\n=== 4. PAPER DIRECTORY ===')
paper_dirs = defaultdict(lambda: [0, 0])
for r in files:
    if r['path'].startswith('paper/'):
        parts = r['path'].split('/')
        sub = parts[1] if len(parts) > 2 else '[root]'
        paper_dirs[sub][0] += 1
        paper_dirs[sub][1] += r['size']

for sub in sorted(paper_dirs.keys()):
    c, sz = paper_dirs[sub]
    print('paper/%-25s | %4d files | %8.2f MB' % (sub, c, sz / (1024 * 1024)))

print('\n=== 5. EXPERIMENTS & CONFIGS & DATA ===')
for top in ['configs', 'experiments', 'data', 'reproducibility', 'logs']:
    subfiles = [r for r in files if r['path'].startswith(top + '/')]
    sz = sum(r['size'] for r in subfiles) / (1024 * 1024)
    print('%-15s | %4d files | %8.2f MB' % (top, len(subfiles), sz))
    for sf in subfiles[:5]:
        print('   - %s (%d KB)' % (sf['path'], sf['size'] // 1024))
    if len(subfiles) > 5:
        print('   ... and %d more files' % (len(subfiles) - 5))
