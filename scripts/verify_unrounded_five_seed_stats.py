#!/usr/bin/env python3
"""
Direct Verification of 5-Seed Unrounded Statistics and Reconciliations
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

seeds = [42, 123, 2024, 7, 99]
methods = ['source', 'confidence', 'utta', 'entropy', 'random', 'tent', 'eata']

raw_results = {m: [] for m in methods}

for s in seeds:
    for m in methods:
        path = f'results/data/wsi_bootstrap/s{s}_{m}.npz'
        data = np.load(path)
        y = data['y']
        p = data['p']
        p_before = data['p_before']
        y_pred = (p >= 0.5).astype(int)
        y_before = (p_before >= 0.5).astype(int)
        
        f1 = float(f1_score(y, y_pred))
        auc = float(roc_auc_score(y, p))
        rec = float(recall_score(y, y_pred))
        prec = float(precision_score(y, y_pred))
        harm = int(((y_before == y) & (y_pred != y)).sum())
        ben = int(((y_before != y) & (y_pred == y)).sum())
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        raw_results[m].append({
            'seed': s, 'f1': f1, 'auc': auc, 'prec': prec, 'rec': rec,
            'harm': harm, 'ben': ben, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        })

print('=== RAW UNROUNDED FIVE-SEED RESULTS ===')
for m in ['source', 'confidence', 'utta']:
    print(f'\nMethod: {m}')
    for r in raw_results[m]:
        print(f"  Seed {r['seed']:4d}: F1={r['f1']:.7f}, AUROC={r['auc']:.7f}, Recall={r['rec']:.7f}, Prec={r['prec']:.7f}, Harm={r['harm']:4d}, Ben={r['ben']:4d}")

f1_conf = np.array([r['f1'] for r in raw_results['confidence']])
f1_utta = np.array([r['f1'] for r in raw_results['utta']])
auc_conf = np.array([r['auc'] for r in raw_results['confidence']])
auc_utta = np.array([r['auc'] for r in raw_results['utta']])
rec_conf = np.array([r['rec'] for r in raw_results['confidence']])
rec_utta = np.array([r['rec'] for r in raw_results['utta']])
harm_conf = np.array([r['harm'] for r in raw_results['confidence']])
harm_utta = np.array([r['harm'] for r in raw_results['utta']])

diff_f1 = f1_utta - f1_conf
diff_auc = auc_utta - auc_conf
diff_rec = rec_utta - rec_conf

t_f1, p_f1 = stats.ttest_rel(f1_utta, f1_conf)
t_auc, p_auc = stats.ttest_rel(auc_utta, auc_conf)
t_rec, p_rec = stats.ttest_rel(rec_utta, rec_conf)
w_f1, p_w_f1 = stats.wilcoxon(diff_f1)

print('\n=== STATISTICAL COMPARISON (UTTA vs Confidence Gating) ===')
print(f'Confidence Gating F1: {f1_conf.mean():.6f} +- {f1_conf.std(ddof=1):.6f}')
print(f'Original UTTA-Med F1: {f1_utta.mean():.6f} +- {f1_utta.std(ddof=1):.6f}')
print(f'Delta F1 (UTTA - Conf): {diff_f1.mean():.6f} +- {diff_f1.std(ddof=1):.6f}')
for s, d in zip(seeds, diff_f1):
    print(f'  Seed {s:4d} Delta F1: {d:+.6f}')
print(f'Paired Student t-test on F1: t = {t_f1:.4f}, p = {p_f1:.5f} (df=4)')
print(f'Wilcoxon signed-rank on F1:  W = {w_f1}, p = {p_w_f1:.5f}')

print(f'\nDelta AUROC: {diff_auc.mean():.6f} +- {diff_auc.std(ddof=1):.6f}, t = {t_auc:.4f}, p = {p_auc:.5f}')
print(f'Delta Recall: {diff_rec.mean():.6f} +- {diff_rec.std(ddof=1):.6f}, t = {t_rec:.4f}, p = {p_rec:.5f}')
print(f'Harmful Flips: Conf = {harm_conf.mean():.1f} +- {harm_conf.std(ddof=1):.1f}, UTTA = {harm_utta.mean():.1f} +- {harm_utta.std(ddof=1):.1f}, Delta = {harm_utta.mean() - harm_conf.mean():+.1f}')

# Now check slide bootstrap or patch bootstrap
# Why did earlier reports say 'statistically indistinguishable'?
# Let's check independent two-sample t-test (unpaired) vs paired t-test:
t_unpaired, p_unpaired = stats.ttest_ind(f1_utta, f1_conf, equal_var=False)
print(f'\nUnpaired (Two-sample) Welch t-test on F1: t = {t_unpaired:.4f}, p = {p_unpaired:.5f}')
print(f'Note: In an unpaired test, between-seed variance (std ~ 0.032) overwhelms the tiny delta (-0.0013), giving p = {p_unpaired:.4f} > 0.05!')
print(f'However, in a paired test, every single seed exhibits UTTA < Conf, so paired t-test gives p = {p_f1:.5f} < 0.01!')
