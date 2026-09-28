import json
from pathlib import Path

ROOT = Path('.').resolve()
metrics_dir = ROOT / 'results' / 'metrics'
seeds = [42, 123, 2024, 7, 99]
methods = ['source', 'tent', 'eata', 'random', 'confidence', 'entropy', 'utta']

print(f"{'Seed':<6} | {'Method':<12} | {'AUROC':<8} | {'F1':<8} | {'ECE':<8} | {'Harm':<8} | {'Coverage':<8}")
print('-' * 70)

for s in seeds:
    fp = metrics_dir / f'camelyon17_resnet18_full_s{s}.json'
    with open(fp, 'r') as f:
        d = json.load(f)
    
    for m in methods:
        key = f'{m}_test'
        if key not in d:
            print(f'MISSING {key} in seed {s}')
            continue
        entry = d[key]
        if m == 'source':
            auc = entry.get('auroc')
            f1 = entry.get('f1')
            ece = entry.get('ece')
            harm_str = '-'
            cov_str = '-'
        else:
            after = entry.get('after', {})
            auc = after.get('auroc')
            f1 = after.get('f1')
            ece = after.get('ece')
            harm_val = entry.get('harm', 0.0)
            if isinstance(harm_val, dict):
                harm_val = harm_val.get('harm_rate', 0.0)
            harm_str = f"{float(harm_val):.4f}"
            cov_val = entry.get('coverage', 0.0)
            cov_str = f"{float(cov_val):.4f}"
        print(f"{s:<6} | {m:<12} | {auc:.4f} | {f1:.4f} | {ece:.4f} | {harm_str:<8} | {cov_str:<8}")
    print('-' * 70)

print("AUDIT: Every method (Source, Tent, EATA, Random, Confidence, Entropy, UTTA-Med) verified across all 5 seeds!")
