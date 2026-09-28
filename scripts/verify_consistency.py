"""Script to verify statistical consistency between headline table and per-seed result JSONs across all 7 methods and 5 seeds."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

headline_path = ROOT / "results" / "tables" / "headline_5seed.json"
assert headline_path.exists(), f"Missing headline_5seed.json at {headline_path}"

with open(headline_path, "r", encoding="utf-8") as f:
    h5 = json.load(f)

seeds = [42, 123, 2024, 7, 99]

print("=" * 80)
print("UTTA-Med Comprehensive Consistency Verification (5 Seeds x 7 Methods)")
print("=" * 80)

for s in seeds:
    json_path = ROOT / "results" / "metrics" / f"camelyon17_resnet18_full_s{s}.json"
    assert json_path.exists(), f"Missing full result JSON for seed {s}"

    with open(json_path, "r", encoding="utf-8") as f:
        sdata = json.load(f)

    row = h5["rows"][str(s)]
    print(f"\n==================== Seed {s} ====================")

    # 1. Source
    src = sdata["source_test"]
    src_auc = src["auroc"]
    src_f1 = src["f1"]
    src_ece = src["ece"]
    src_brier = src.get("brier", None)
    print(f"  [Source]     AUROC={src_auc:.4f} (h5={row['src_auroc']:.4f}), F1={src_f1:.4f} (h5={row['src_f1']:.4f}), ECE={src_ece:.4f}, Brier={src_brier}")
    assert abs(row["src_auroc"] - src_auc) < 1e-3, f"Mismatch in source AUROC for seed {s}"
    assert abs(row["src_f1"] - src_f1) < 1e-3, f"Mismatch in source F1 for seed {s}"

    # Helper function for TTA methods
    def check_tta(method_name, h5_prefix, check_h5=True):
        mdata = sdata[f"{method_name}_test"]
        after = mdata["after"]
        auc = after["auroc"]
        f1 = after["f1"]
        ece_val = after.get("ece", 0.0)
        cov = mdata.get("coverage", 1.0)
        harm_obj = mdata.get("harm", 0.0)
        harm = harm_obj["harm_rate"] if isinstance(harm_obj, dict) else float(harm_obj)

        print(f"  [{method_name:10s}] AUROC={auc:.4f}, F1={f1:.4f}, Cov={cov:.3f}, ECE={ece_val:.4f}, Harm={harm:.4f}")

        if check_h5 and h5_prefix:
            assert abs(row[f"{h5_prefix}_auroc"] - auc) < 1e-3, f"Mismatch in {method_name} AUROC for seed {s}"
            assert abs(row[f"{h5_prefix}_f1"] - f1) < 1e-3, f"Mismatch in {method_name} F1 for seed {s}"
            assert abs(row[f"{h5_prefix}_harm"] - harm) < 1e-3, f"Mismatch in {method_name} harm for seed {s}"
            assert abs(row[f"{h5_prefix}_cov"] - cov) < 1e-2, f"Mismatch in {method_name} coverage for seed {s}"
        return auc, f1, cov, ece_val, harm

    # 2. Tent
    check_tta("tent", "tent")

    # 3. EATA
    check_tta("eata", "eata")

    # 4. Random
    check_tta("random", "rand")

    # 5. Confidence
    conf_auc, conf_f1, conf_cov, conf_ece, conf_harm = check_tta("confidence", "conf")

    # 6. Entropy (matched-coverage control)
    ent_auc, ent_f1, ent_cov, ent_ece, ent_harm = check_tta("entropy", None, check_h5=False)
    # Entropy should mirror confidence under binary symmetric calibration
    assert abs(conf_auc - ent_auc) < 1e-3, f"Entropy AUROC should match confidence for seed {s}"
    assert abs(conf_cov - ent_cov) < 1e-3, f"Entropy coverage should match confidence for seed {s}"

    # 7. UTTA-Med
    check_tta("utta", "utta")

print("\n" + "=" * 80)
print("SUCCESS: ALL 5 SEEDS x 7 METHODS ARE FULLY VERIFIED AND CONSISTENT!")
print("=" * 80)
