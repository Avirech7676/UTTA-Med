import json

with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

files = data['files']
print("=== ALL FILES IN PAPER/IEEE ===")
for r in files:
    if r['path'].startswith('paper/ieee/') and not r['path'].startswith('paper/ieee/figs/'):
        print("%-55s | %6.1f KB | %s" % (r['path'], r['size']/1024, r['mtime']))
