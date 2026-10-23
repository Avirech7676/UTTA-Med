import pandas as pd
import numpy as np
import scipy.stats as stats

def run_audit():
    print("=== TASK 1: STATISTICAL TEST AUDIT ===")
    df = pd.read_csv('10_BIO_SAFEADAPT_ABLATION_RESULTS.csv')
    
    # Filter for seeds (numeric only)
    df['seed_num'] = pd.to_numeric(df['seed'], errors='coerce')
    df_seeds = df.dropna(subset=['seed_num']).copy()
    df_seeds['seed'] = df_seeds['seed_num'].astype(int)
    
    d0 = df_seeds[df_seeds['method_id'] == 'D0_Frozen_Source'].sort_values('seed')
    d1 = df_seeds[df_seeds['method_id'] == 'D1_Original_SafeAdapt'].sort_values('seed')
    d8 = df_seeds[df_seeds['method_id'] == 'D8_Cellularity_Anchoring_Resets'].sort_values('seed')
    d9 = df_seeds[df_seeds['method_id'] == 'D9_Complete_Bio_SafeAdapt'].sort_values('seed')
    
    print("D1 F1s:", d1['f1_score'].values)
    print("D8 F1s:", d8['f1_score'].values)
    print("D9 F1s:", d9['f1_score'].values)
    print("D0 F1s:", d0['f1_score'].values)
    
    # Paired differences: D9 - D1
    diffs_9_1 = d9['f1_score'].values - d1['f1_score'].values
    print("D9 - D1 diffs:", diffs_9_1)
    
    mean_diff = np.mean(diffs_9_1)
    std_diff = np.std(diffs_9_1, ddof=1)
    se_diff = std_diff / np.sqrt(len(diffs_9_1))
    t_stat = mean_diff / se_diff
    df_t = len(diffs_9_1) - 1
    
    # Two-sided p-value
    p_two_sided = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_t))
    # One-sided p-value (H1: mean_diff > 0)
    p_one_sided = 1 - stats.t.cdf(t_stat, df=df_t)
    
    # 95% Confidence Interval
    t_crit = stats.t.ppf(0.975, df=df_t)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # Cohen's dz
    cohens_d = mean_diff / std_diff
    
    print(f"Paired Student t-test: D9 vs D1 (n={len(diffs_9_1)}):")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Std diff (ddof=1): {std_diff:.6f}")
    print(f"  SE diff: {se_diff:.6f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  df: {df_t}")
    print(f"  p-value (two-sided): {p_two_sided:.6f} (approx {p_two_sided:.4f})")
    print(f"  p-value (one-sided): {p_one_sided:.6f} (approx {p_one_sided:.4f})")
    print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}] (rounded [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"  Cohen's d_z: {cohens_d:.4f}")
    
    print("\n=== TASK 2: D8 vs D9 COMPARISON ===")
    cols_to_check = ['seed', 'true_positives', 'false_positives', 'false_negatives', 
                     'precision', 'recall', 'f1_score', 'admitted_updates', 
                     'rollbacks', 'permanent_fallbacks', 'false_silence_alarms', 
                     'gpu_inference_time_ms', 'peak_gpu_memory_mb']
    
    print("\nD8 rows:")
    print(d8[cols_to_check].to_string(index=False))
    
    print("\nD9 rows:")
    print(d9[cols_to_check].to_string(index=False))
    
    diff_d8_d9 = {}
    for col in cols_to_check:
        if col != 'seed':
            diff_d8_d9[col] = d9[col].values - d8[col].values
            print(f"Difference D9 - D8 for {col}: {diff_d8_d9[col]}")

if __name__ == '__main__':
    run_audit()
