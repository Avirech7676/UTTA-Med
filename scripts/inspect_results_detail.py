import json
from collections import defaultdict

with open('results/workspace_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

files = data['files']

print("=== RESULTS SUBDIRECTORIES IN DETAIL ===")
results_files = [r for r in files if r['path'].startswith('results/')]
sub_breakdown = defaultdict(list)
for r in results_files:
    parts = r['path'].split('/')
    sub = parts[1] if len(parts) > 2 else '[root]'
    sub_breakdown[sub].append(r)

for sub in sorted(sub_breakdown.keys()):
    items = sub_breakdown[sub]
    total_sz = sum(x['size'] for x in items) / (1024 * 1024)
    print(f"\n--- results/{sub} ({len(items)} files, {total_sz:.2f} MB) ---")
    # print up to 8 sample items
    for item in items[:8]:
        print("  %-60s | %6.1f KB" % (item['path'], item['size'] / 1024))
    if len(items) > 8:
        print(f"  ... and {len(items)-8} more files")
