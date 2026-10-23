#!/usr/bin/env python3
"""
Generate BIO_SAFEADAPT_VERIFIED_PREDICTIONS.csv
Independent 1:1 Euclidean Centroid Matching on MIDOG 2021 genuine test patches.
Evaluates:
  1. Frozen Faster R-CNN (Arm A)
  2. Original SafeAdapt (Arm D)
  3. Bio-SafeAdapt Controller (Arm E)
Across seeds 42, 7, 123.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_verification():
    raw_path = ROOT / "results/generalization/h3/h3_2_r3b/input_manifests/raw_predictions.csv"
    gt_path = ROOT / "results/generalization/h3/h3_2_r3b/input_manifests/ground_truth_annotations.csv"
    manifest_path = ROOT / "results/generalization/h3/h3_2_r3b/input_manifests/target_patch_manifest.csv"

    print(f"Loading raw predictions from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    gt_df = pd.read_csv(gt_path)
    manifest_df = pd.read_csv(manifest_path)

    arm_names = {
        "Arm_A": "Frozen_Source_FasterRCNN",
        "Arm_D": "Original_SafeAdapt",
        "Arm_E": "Bio_SafeAdapt_Controller"
    }

    # Filter boundary slits (is_sub_1px == False)
    valid_raw = raw_df[~raw_df["is_sub_1px"] & raw_df["arm"].isin(arm_names.keys())].copy()

    # Pre-build patch metadata map by g_pid (0..1099)
    patch_meta = {}
    for _, row in manifest_df.iterrows():
        cid = int(row["case_id"])
        pidx = int(row["patch_index"])
        g_pid = (cid - 101) * 22 + pidx
        patch_meta[g_pid] = {
            "case_id": cid,
            "patch_index_local": pidx,
            "origin_x": float(row["origin_x"]),
            "origin_y": float(row["origin_y"])
        }

    radius_px = 30.0
    conf_thresh = 0.20

    # Build GT by global_patch_id (0..1099)
    gt_by_gpid = {i: [] for i in range(1100)}
    for idx, row in gt_df.iterrows():
        cid = int(row["case_id"])
        pidx = int(row["patch_index"])
        g_pid = (cid - 101) * 22 + pidx
        gt_by_gpid[g_pid].append({
            "gt_id": idx,
            "case_id": cid,
            "patch_index": pidx,
            "global_x": float(row["global_x"]),
            "global_y": float(row["global_y"]),
            "local_x": float(row["local_x"]),
            "local_y": float(row["local_y"])
        })

    records = []
    summary_metrics = []

    seeds = [42, 7, 123]
    for seed in seeds:
        for arm_code, method_name in arm_names.items():
            sub = valid_raw[(valid_raw["seed"] == seed) & (valid_raw["arm"] == arm_code)]
            
            # Group predictions by g_pid
            preds_by_gpid = {i: [] for i in range(1100)}
            for _, r in sub.iterrows():
                g_pid = int(r["patch_index"])
                c_loc = json.loads(r["centroid_local"]) if isinstance(r["centroid_local"], str) else list(r["centroid_local"])
                c_glob = json.loads(r["centroid_global"]) if isinstance(r["centroid_global"], str) else list(r["centroid_global"])
                preds_by_gpid[g_pid].append({
                    "score": float(r["score"]),
                    "box_local": r["box_local"],
                    "centroid_local": c_loc,
                    "centroid_global": c_glob,
                    "width_px": float(r["width_px"]),
                    "height_px": float(r["height_px"]),
                    "origin_x": float(r["patch_origin_x"]),
                    "origin_y": float(r["patch_origin_y"]),
                    "case_id": int(r["case_id"]),
                    "patch_index": g_pid
                })

            tp_total, fp_total, fn_total = 0, 0, 0
            
            for g_pid in range(1100):
                gts = gt_by_gpid[g_pid]
                preds = preds_by_gpid[g_pid]
                meta = patch_meta.get(g_pid, {"case_id": 101 + g_pid // 22, "patch_index_local": g_pid % 22, "origin_x": 0.0, "origin_y": 0.0})

                # Sort preds by score descending
                preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
                matched_gt_indices = set()

                for p in preds_sorted:
                    score = p["score"]
                    px, py = p["centroid_local"]
                    
                    if score < conf_thresh:
                        records.append({
                            "seed": seed,
                            "arm": arm_code,
                            "method": method_name,
                            "case_id": meta["case_id"],
                            "global_patch_index": g_pid,
                            "local_patch_index": meta["patch_index_local"],
                            "patch_origin_x": p["origin_x"],
                            "patch_origin_y": p["origin_y"],
                            "centroid_local_x": round(px, 2),
                            "centroid_local_y": round(py, 2),
                            "centroid_global_x": round(p["centroid_global"][0], 2),
                            "centroid_global_y": round(p["centroid_global"][1], 2),
                            "score": round(score, 4),
                            "match_status": "BELOW_THRESHOLD",
                            "matched_gt_id": -1,
                            "matched_gt_distance_px": np.nan
                        })
                        continue

                    # Candidates within radius
                    best_dist = 1e9
                    best_gt_idx = None
                    for g_idx, g in enumerate(gts):
                        if g_idx in matched_gt_indices:
                            continue
                        dist = np.hypot(px - g["local_x"], py - g["local_y"])
                        if dist <= radius_px and dist < best_dist:
                            best_dist = dist
                            best_gt_idx = g_idx

                    if best_gt_idx is not None:
                        matched_gt_indices.add(best_gt_idx)
                        tp_total += 1
                        records.append({
                            "seed": seed,
                            "arm": arm_code,
                            "method": method_name,
                            "case_id": meta["case_id"],
                            "global_patch_index": g_pid,
                            "local_patch_index": meta["patch_index_local"],
                            "patch_origin_x": p["origin_x"],
                            "patch_origin_y": p["origin_y"],
                            "centroid_local_x": round(px, 2),
                            "centroid_local_y": round(py, 2),
                            "centroid_global_x": round(p["centroid_global"][0], 2),
                            "centroid_global_y": round(p["centroid_global"][1], 2),
                            "score": round(score, 4),
                            "match_status": "TRUE_POSITIVE",
                            "matched_gt_id": gts[best_gt_idx]["gt_id"],
                            "matched_gt_distance_px": round(best_dist, 2)
                        })
                    else:
                        fp_total += 1
                        records.append({
                            "seed": seed,
                            "arm": arm_code,
                            "method": method_name,
                            "case_id": meta["case_id"],
                            "global_patch_index": g_pid,
                            "local_patch_index": meta["patch_index_local"],
                            "patch_origin_x": p["origin_x"],
                            "patch_origin_y": p["origin_y"],
                            "centroid_local_x": round(px, 2),
                            "centroid_local_y": round(py, 2),
                            "centroid_global_x": round(p["centroid_global"][0], 2),
                            "centroid_global_y": round(p["centroid_global"][1], 2),
                            "score": round(score, 4),
                            "match_status": "FALSE_POSITIVE",
                            "matched_gt_id": -1,
                            "matched_gt_distance_px": np.nan
                        })

                # False Negatives
                for g_idx, g in enumerate(gts):
                    if g_idx not in matched_gt_indices:
                        fn_total += 1
                        records.append({
                            "seed": seed,
                            "arm": arm_code,
                            "method": method_name,
                            "case_id": meta["case_id"],
                            "global_patch_index": g_pid,
                            "local_patch_index": meta["patch_index_local"],
                            "patch_origin_x": meta["origin_x"],
                            "patch_origin_y": meta["origin_y"],
                            "centroid_local_x": round(g["local_x"], 2),
                            "centroid_local_y": round(g["local_y"], 2),
                            "centroid_global_x": round(g["global_x"], 2),
                            "centroid_global_y": round(g["global_y"], 2),
                            "score": 1.0000,
                            "match_status": "FALSE_NEGATIVE",
                            "matched_gt_id": g["gt_id"],
                            "matched_gt_distance_px": 0.00
                        })

            prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            summary_metrics.append({
                "seed": seed,
                "arm": arm_code,
                "method": method_name,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            })
            print(f"Seed {seed:<3} | {method_name:<26} | TP={tp_total:>2}, FP={fp_total:>3}, FN={fn_total:>2} | Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    out_df = pd.DataFrame(records)
    out_csv = ROOT / "BIO_SAFEADAPT_VERIFIED_PREDICTIONS.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[+] Wrote: {out_csv.name} ({len(out_df)} records)")

    summary_df = pd.DataFrame(summary_metrics)
    print("\n" + "=" * 80)
    print("INDEPENDENT RECONSTRUCTION SUMMARY:")
    print("=" * 80)
    print(summary_df.to_string())

if __name__ == "__main__":
    run_verification()
