import json
import os
import hashlib
import pandas as pd
import numpy as np

def sha(p):
    if not os.path.exists(p):
        return 'MISSING'
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

# Load all authoritative artifacts
h5_path = 'results/tables/headline_5seed.json'
paired_path = 'results/tables/table_paired_statistics_final.csv'
main_path = 'results/tables/table_main_results_final.csv'
d2_path = 'results/v2/stage_d2/stage_d2_benchmark_results.json'
e0_path = 'results/v2/stage_e0/stage_e0_reconciliation.json'
e1_path = 'results/v2/stage_e1/stage_e1_evaluation_results.json'
e2_path = 'results/v2/stage_e2/stage_e2_target_results.json'

with open(h5_path) as f: h5 = json.load(f)
df_paired = pd.read_csv(paired_path)
df_main = pd.read_csv(main_path)
with open(d2_path) as f: d2 = json.load(f)
with open(e0_path) as f: e0 = json.load(f)
with open(e1_path) as f: e1 = json.load(f)
with open(e2_path) as f: e2 = json.load(f)

def df_to_md(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)

print("ALL ARTIFACTS LOADED SUCCESSFULLY")

# Print Table 1: V1 5-seed performance
print("\n" + "="*80)
print("TABLE 1: V1 FIVE-SEED PERFORMANCE")
print("="*80)
print(df_to_md(df_main))

# Print Table 2: V1 Paired Statistics
print("\n" + "="*80)
print("TABLE 2: V1 PAIRED STATISTICAL COMPARISONS")
print("="*80)
print(df_to_md(df_paired))


# Print Table 3: D2 Gating Comparison (Experiment A & D)
print("\n" + "="*80)
print("TABLE 3: STAGE D2 GATING COMPARISON (HOSPITAL 1)")
print("="*80)
exp_a = d2['experiment_a_natural_policies']
exp_d = d2['experiment_d_efficiency_benchmark']
rows_d2 = []
for name, data in exp_a.items():
    m = data['metrics']
    t = data['telemetry']
    runtime_s = t.get('total_wall_clock_s', np.nan)
    cov = t.get('patch_coverage', np.nan)
    mc_passes = t.get('total_mc_passes_executed', np.nan)
    rows_d2.append({
        'Policy': name,
        'AUROC': f"{m['auroc']:.4f}",
        'AUPRC': f"{m['auprc']:.4f}",
        'F1': f"{m['f1']:.4f}",
        'ECE': f"{m['ece']:.4f}",
        'Brier': f"{m['brier']:.6f}",
        'Patch Coverage (%)': f"{cov*100:.1f}%",
        'MC Passes': mc_passes,
        'Wall Clock (s)': f"{runtime_s:.2f}"
    })
df_t3 = pd.DataFrame(rows_d2)
print(df_to_md(df_t3))

# Print Table 4: E1 Hospital-1 4-Arm Evaluation
print("\n" + "="*80)
print("TABLE 4: STAGE E1 FOUR-ARM EVALUATION (HOSPITAL 1, 320 PATCHES, PRE-ADAPTATION)")
print("="*80)
e1_arms = e1['primary_experiment_wsi_reset']
rows_e1 = []
for arm_id, arm_data in e1_arms.items():
    m = arm_data['overall_metrics']
    rows_e1.append({
        'Arm ID': arm_id.upper(),
        'Arm Description': arm_data['arm_name'],
        'Macro Valid AUROC': f"{m['macro_auroc_valid_slides']:.4f}",
        'Pooled AUROC': f"{m['auroc']:.4f}",
        'Pooled AUPRC': f"{m['auprc']:.4f}",
        'Pooled F1': f"{m['f1']:.4f}",
        'ECE': f"{m['ece']:.4f}",
        'Brier': f"{m['brier']:.6f}",
        'Elapsed (s)': f"{arm_data['elapsed_s']:.2f}"
    })
df_t4 = pd.DataFrame(rows_e1)
print(df_to_md(df_t4))

# Print Table 5: E2 Hospital-2 4-Arm Evaluation
print("\n" + "="*80)
print("TABLE 5: STAGE E2 FOUR-ARM TARGET EVALUATION (HOSPITAL 2, 1280 PATCHES, PRE-ADAPTATION)")
print("="*80)
e2_arms = e2['arms']
rows_e2 = []
for arm_id, arm_data in e2_arms.items():
    m = arm_data['overall_metrics']
    rows_e2.append({
        'Arm ID': arm_id.upper(),
        'Arm Description': arm_data['arm_name'],
        'Macro Valid AUROC': f"{m['macro_auroc_valid_slides']:.4f}",
        'Pooled AUROC': f"{m['auroc']:.4f}",
        'Pooled AUPRC': f"{m['auprc']:.4f}",
        'Pooled F1': f"{m['f1']:.4f}",
        'ECE': f"{m['ece']:.4f}",
        'Brier': f"{m['brier']:.6f}",
        'Elapsed (s)': f"{arm_data['elapsed_s']:.2f}"
    })
df_t5 = pd.DataFrame(rows_e2)
print(df_to_md(df_t5))

# Print Table 6: Per-WSI Performance on Hospital 2 (Arms A, B, C, D)
print("\n" + "="*80)
print("TABLE 6: PER-WSI TARGET PERFORMANCE (HOSPITAL 2, WSIs 20-29)")
print("="*80)
slides = [str(i) for i in range(20, 30)]
rows_wsi = []
print("Slide 20 keys:", list(e2_arms['arm_a']['slide_metrics']['20'].keys()))
def fmt_val(v, decimals=4):
    if v is None:
        return "N/A (single-class)"
    try:
        val = float(v)
        if np.isnan(val):
            return "N/A (single-class)"
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)

for s in slides:
    s_a = e2_arms['arm_a']['slide_metrics'][s]
    s_b = e2_arms['arm_b']['slide_metrics'][s]
    s_c = e2_arms['arm_c']['slide_metrics'][s]
    s_d = e2_arms['arm_d']['slide_metrics'][s]
    rows_wsi.append({
        'WSI': f"slide_{s}",
        'N': s_a.get('n', 128),
        'Arm A AUROC': fmt_val(s_a.get('auroc')),
        'Arm B AUROC': fmt_val(s_b.get('auroc')),
        'Arm D AUROC': fmt_val(s_d.get('auroc')),
        'Arm A F1': fmt_val(s_a.get('f1')),
        'Arm B F1': fmt_val(s_b.get('f1')),
        'Arm D F1': fmt_val(s_d.get('f1')),
        'Arm A Brier': fmt_val(s_a.get('brier')),
        'Arm B Brier': fmt_val(s_b.get('brier')),
        'Arm D Brier': fmt_val(s_d.get('brier'))
    })

df_t6 = pd.DataFrame(rows_wsi)
print(df_to_md(df_t6))


# Print Table 7: Safeguard intervention counts
print("\n" + "="*80)
print("TABLE 7: SAFEGUARD INTERVENTION COUNTS AND TELEMETRY")
print("="*80)
rows_t7 = []
for stage_name, stage_data, arm_keys in [
    ('Stage E1 (Hospital 1)', e1['primary_experiment_wsi_reset'], ['arm_a', 'arm_b', 'arm_c', 'arm_d']),
    ('Stage E2 (Hospital 2)', e2['arms'], ['arm_a', 'arm_b', 'arm_c', 'arm_d'])
]:
    for a_id in arm_keys:
        a_data = stage_data[a_id]
        tel = a_data['slide_telemetry']
        first_tel = next(iter(tel.values())) if tel else {}
        print(f"Tel keys for {a_id}:", list(first_tel.keys()))
        tot_batches = sum(t.get('total_batches', t.get('batches', 0)) for t in tel.values())
        tot_eligible = sum(t.get('total_eligible_patches', t.get('eligible_patches', 0)) for t in tel.values())
        tot_selected = sum(t.get('total_selected_patches', t.get('selected_patches', 0)) for t in tel.values())
        tot_attempted = sum(t.get('candidate_updates', 0) for t in tel.values())
        tot_accepted = sum(t.get('accepted_updates', 0) for t in tel.values())
        tot_rejected = sum(t.get('rejected_updates', 0) for t in tel.values())
        tot_selected = sum(t.get('selected_patches', 0) for t in tel.values())
        nan_inf = sum(t.get('non_finite_failures', 0) for t in tel.values())
        disp_viol = sum(t.get('param_bound_clamps', 0) for t in tel.values())
        eligible_mon = sum(t.get('eligible_monitor_decisions', 0) for t in tel.values())
        alarms = sum(t.get('monitor_warnings', 0) for t in tel.values())
        rollbacks = sum(t.get('rollbacks', 0) for t in tel.values())
        fallbacks = sum(t.get('fallbacks', 0) for t in tel.values())
        max_disp = max([t.get('max_displacement_linf', 0.0) for t in tel.values()] or [0.0])
        
        rows_t7.append({
            'Evaluation': stage_name,
            'Arm': a_id.upper(),
            'Selected Patches': tot_selected,
            'Updates Attempted': tot_attempted,
            'Updates Accepted': tot_accepted,
            'Updates Rejected': tot_rejected,
            'Eligible Decisions': eligible_mon,
            'Alarms Triggered': alarms,
            'Rollbacks Executed': rollbacks,
            'Fallbacks': fallbacks,
            'NaN/Inf Events': nan_inf,
            'Bound Clamps': disp_viol,
            'Max Displacement (L_inf)': f"{max_disp:.2e}"
        })

df_t7 = pd.DataFrame(rows_t7)
print(df_to_md(df_t7))

# Print Table 8: Runtime and computational overhead
print("\n" + "="*80)
print("TABLE 8: RUNTIME AND COMPUTATIONAL OVERHEAD")
print("="*80)
rows_t8 = [
    {
        'Benchmark / Cohort': 'Stage D2 (Hospital 1, 320 patches, 20 MC samples)',
        'Frozen Source': f"{d2['experiment_d_efficiency_benchmark']['runtimes_s']['frozen_source']:.2f}s",
        'Confidence-Only': f"{d2['experiment_d_efficiency_benchmark']['runtimes_s']['confidence_only']:.2f}s",
        'Hybrid': f"{d2['experiment_d_efficiency_benchmark']['runtimes_s']['hybrid']:.2f}s",
        'MC-Dropout Only': f"{d2['experiment_d_efficiency_benchmark']['runtimes_s']['mc_only']:.2f}s",
        'Active SafeAdapt': 'N/A'
    },
    {
        'Benchmark / Cohort': 'Stage E1 (Hospital 1, 320 patches, Pre-Adaptation)',
        'Frozen Source': f"{e1['primary_experiment_wsi_reset']['arm_a']['elapsed_s']:.2f}s",
        'Confidence-Only': f"{e1['primary_experiment_wsi_reset']['arm_b']['elapsed_s']:.2f}s",
        'Hybrid': 'N/A',
        'MC-Dropout Only': 'N/A',
        'Active SafeAdapt': f"{e1['primary_experiment_wsi_reset']['arm_d']['elapsed_s']:.2f}s"
    },
    {
        'Benchmark / Cohort': 'Stage E2 (Hospital 2, 1280 patches, Pre-Adaptation)',
        'Frozen Source': f"{e2['arms']['arm_a']['elapsed_s']:.2f}s",
        'Confidence-Only': f"{e2['arms']['arm_b']['elapsed_s']:.2f}s",
        'Hybrid': 'N/A',
        'MC-Dropout Only': 'N/A',
        'Active SafeAdapt': f"{e2['arms']['arm_d']['elapsed_s']:.2f}s"
    }
]
df_t8 = pd.DataFrame(rows_t8)
print(df_to_md(df_t8))

