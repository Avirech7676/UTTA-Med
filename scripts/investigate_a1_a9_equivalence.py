#!/usr/bin/env python3
"""Phase 1: Exhaustive Investigation of A1 (Confidence Gating) and A9 (UTTA-Med 2.0) Equivalence.

Performs:
1. Forensic audit of saved predictions, npz files, and benchmark tables across 5 seeds.
2. Direct comparison of raw prediction probabilities, labels, and coverage.
3. Code-level audit of AdaptiveUncertaintyGating and ConfidenceGatingPolicy.
4. Live evaluation on real Camelyon17 patches to test component activation,
   sample admission/rejection, uncertainty distribution, and parameter updates.
5. Detection of reused metric files and cached predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image

from src.models.resnet18 import build_resnet18
from src.v2.gating.adaptive_uncertainty import AdaptiveUncertaintyGating
from src.v2.gating.policies import ConfidenceGatingPolicy
from src.v2.gating.multi_signal import MultiSignalGatingPolicy
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.adapter import SafeAdaptTTA, binary_entropy_from_logits
from src.v2.safeadapt.controller import SafeAdaptController

SEEDS = [42, 123, 2024, 7, 99]

def audit_saved_artifacts():
    print("=" * 80)
    print("STAGE 1: FORENSIC AUDIT OF SAVED PREDICTION FILES & BENCHMARK TABLES")
    print("=" * 80)

    bootstrap_dir = ROOT / "results" / "data" / "wsi_bootstrap"
    print(f"[*] Checking {bootstrap_dir}...")
    npz_files = list(bootstrap_dir.glob("*.npz"))
    print(f"[*] Total files in wsi_bootstrap: {len(npz_files)}")
    
    file_names = [f.name for f in npz_files]
    for s in SEEDS:
        has_conf = f"s{s}_confidence.npz" in file_names
        has_utta = f"s{s}_utta.npz" in file_names
        has_src = f"s{s}_source.npz" in file_names
        has_a9 = f"s{s}_a9.npz" in file_names or f"s{s}_complete.npz" in file_names
        print(f"  Seed {s:4d}: s{s}_confidence={has_conf}, s{s}_utta={has_utta}, s{s}_source={has_src}, s{s}_a9={has_a9}")

    headline_path = ROOT / "results" / "tables" / "headline_5seed.json"
    if headline_path.exists():
        with open(headline_path) as f:
            headline = json.load(f)
        print(f"[*] headline_5seed.json keys: {list(headline.keys())}")
        rows = headline.get("rows", {})
        print(f"[*] headline_5seed.json contains seeds: {list(rows.keys())}")
        for s, r in rows.items():
            print(f"  Seed {s}: conf_f1={r.get('conf_f1')}, utta_f1={r.get('utta_f1')}, conf_cov={r.get('conf_cov')}")

    ablation_csv = ROOT / "MULTI_SIGNAL_ABLATION_RESULTS.csv"
    if ablation_csv.exists():
        df_abl = pd.read_csv(ablation_csv)
        print(f"[*] MULTI_SIGNAL_ABLATION_RESULTS.csv has {len(df_abl)} rows.")
        a1_df = df_abl[df_abl["ablation_arm"] == "A1_confidence_gating"].set_index("seed")
        a9_df = df_abl[df_abl["ablation_arm"] == "A9_complete_utta_med_2_0"].set_index("seed")
        print("\n[*] Comparing A1 vs A9 values in MULTI_SIGNAL_ABLATION_RESULTS.csv:")
        for s in SEEDS:
            if s in a1_df.index and s in a9_df.index:
                r1 = a1_df.loc[s]
                r9 = a9_df.loc[s]
                print(f"  Seed {s:4d}: A1 AUROC={r1['auroc']}, F1={r1['binary_f1']}, Cov={r1['adaptation_coverage']} | "
                      f"A9 AUROC={r9['auroc']}, F1={r9['binary_f1']}, Cov={r9['adaptation_coverage']}")

def load_real_validation_sample(n_samples: int = 200, seed: int = 42):
    parquet_path = Path("c:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data/validation-00002-of-00003.parquet")
    pf = pq.ParquetFile(parquet_path)
    tbl_0 = pf.read_row_group(0, columns=["image", "label"])
    tbl_1 = pf.read_row_group(115, columns=["image", "label"])
    
    combined = []
    for tbl in [tbl_0, tbl_1]:
        df = tbl.to_pandas()
        for _, row in df.iterrows():
            lbl = int(row["label"])
            b = row["image"]["bytes"]
            im = Image.open(io.BytesIO(b)).convert("RGB")
            arr = (np.array(im, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            combined.append((torch.from_numpy(arr).permute(2, 0, 1).float(), lbl))
            if len(combined) >= n_samples:
                break
        if len(combined) >= n_samples:
            break
            
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    imgs = torch.stack([item[0] for item in combined], dim=0)
    labels = np.array([item[1] for item in combined])
    return imgs, labels

def audit_controller_and_gating_behavior():
    print("\n" + "=" * 80, flush=True)
    print("STAGE 2: ALGORITHMIC COMPARISON OF A1 AND A9 POLICIES ON REAL PATCHES", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cpu")
    x, y_true = load_real_validation_sample(n_samples=64, seed=42)
    print(f"[*] Loaded {len(x)} real Camelyon17 validation patches.", flush=True)

    centroids_path = ROOT / "checkpoints" / "source_centroids_resnet18_s42.pt"
    centroids_data = torch.load(centroids_path, map_location=device, weights_only=False)
    centroids = centroids_data["centroids"] if "centroids" in centroids_data else centroids_data["prototypes"]

    results = []

    for s in SEEDS:
        ckpt_path = ROOT / "checkpoints" / f"camelyon17_resnet18_source_s{s}_BEST.pt"
        if not ckpt_path.exists():
            print(f"[!] Checkpoint {ckpt_path.name} not found, skipping seed {s}")
            continue

        model_a1 = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
        model_a9 = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model_a1.load_state_dict(state_dict)
        model_a9.load_state_dict(state_dict)

        policy_a1 = ConfidenceGatingPolicy(tau_conf=0.90)
        policy_a9 = AdaptiveUncertaintyGating(
            tau_high=0.90,
            tau_low=0.65,
            tau_unc=8.875383e-06,
            max_mc_passes=20,
            min_mc_passes=5,
            chunk_size=5,
            use_consistency=True,
            consistency_threshold=0.05,
            source_centroids=centroids,
            seed=s
        )

        dec_a1 = policy_a1.select(model_a1, x)
        dec_a9 = policy_a9.select(model_a9, x)

        mask_a1 = dec_a1.keep_mask.cpu().numpy()
        mask_a9 = dec_a9.keep_mask.cpu().numpy()

        admit_agreement = float(np.mean(mask_a1 == mask_a9)) * 100.0
        n_admit_a1 = int(mask_a1.sum())
        n_admit_a9 = int(mask_a9.sum())
        n_differ = int(np.sum(mask_a1 != mask_a9))

        with torch.no_grad():
            model_a1.eval()
            model_a9.eval()
            p_a1 = torch.sigmoid(model_a1(x)).view(-1).cpu().numpy()
            p_a9 = torch.sigmoid(model_a9(x)).view(-1).cpu().numpy()

        prob_abs_diff = float(np.mean(np.abs(p_a1 - p_a9)))
        labels_differ = int(np.sum((p_a1 >= 0.5) != (p_a9 >= 0.5)))
        identical_preds = float(np.mean(p_a1 == p_a9)) * 100.0

        # Configure BatchNorm affine adaptation (TENT style)
        for m in model_a1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
            else:
                m.requires_grad_(False)
        for m in model_a9.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
            else:
                m.requires_grad_(False)

        params_a1 = [p for p in model_a1.parameters() if p.requires_grad]
        params_a9 = [p for p in model_a9.parameters() if p.requires_grad]

        opt_a1 = torch.optim.Adam(params_a1, lr=1e-4)
        opt_a9 = torch.optim.Adam(params_a9, lr=1e-4)

        orig_params_a1 = [p.clone().detach() for p in params_a1]
        orig_params_a9 = [p.clone().detach() for p in params_a9]

        # Adapt A1
        if dec_a1.keep_mask.sum() > 0:
            model_a1.train()
            for m in model_a1.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
                    m.eval()
            out_a1 = model_a1(x[dec_a1.keep_mask])
            p_ad_a1 = torch.sigmoid(out_a1).clamp(1e-6, 1.0 - 1e-6)
            loss_a1 = -(p_ad_a1 * torch.log(p_ad_a1) + (1.0 - p_ad_a1) * torch.log(1.0 - p_ad_a1)).mean()
            loss_a1.backward()
            opt_a1.step()
            opt_a1.zero_grad()

        # Adapt A9
        if dec_a9.keep_mask.sum() > 0:
            model_a9.train()
            for m in model_a9.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
                    m.eval()
            out_a9 = model_a9(x[dec_a9.keep_mask])
            p_ad_a9 = torch.sigmoid(out_a9).clamp(1e-6, 1.0 - 1e-6)
            loss_a9 = -(p_ad_a9 * torch.log(p_ad_a9) + (1.0 - p_ad_a9) * torch.log(1.0 - p_ad_a9)).mean()
            loss_a9.backward()
            opt_a9.step()
            opt_a9.zero_grad()

        norm_diff_a1 = sum(torch.norm(p - orig).item() for p, orig in zip(params_a1, orig_params_a1))
        norm_diff_a9 = sum(torch.norm(p - orig).item() for p, orig in zip(params_a9, orig_params_a9))
        param_divergence_a1_a9 = sum(torch.norm(p1 - p9).item() for p1, p9 in zip(params_a1, params_a9))

        rec = {
            "seed": s,
            "admit_agreement_pct": admit_agreement,
            "n_admit_a1": n_admit_a1,
            "n_admit_a9": n_admit_a9,
            "n_differ_decisions": n_differ,
            "prob_abs_diff": prob_abs_diff,
            "labels_differ": labels_differ,
            "identical_preds_pct": identical_preds,
            "param_update_norm_a1": norm_diff_a1,
            "param_update_norm_a9": norm_diff_a9,
            "param_divergence_a1_vs_a9": param_divergence_a1_a9,
            "a9_routed_to_mc": int(dec_a9.routed_to_mc.sum().item()),
            "a9_mc_passes": dec_a9.mc_passes_executed,
            "a9_confident_accepted": policy_a9.telemetry.confident_auto_accepted,
            "a9_low_conf_rejected": policy_a9.telemetry.low_conf_auto_rejected,
            "a9_mc_accepted": int(mask_a9.sum()) - policy_a9.telemetry.confident_auto_accepted,
        }
        results.append(rec)
        print(f"\n--- SEED {s} ---")
        print(f"  Admission Agreement: {admit_agreement:.2f}% (Differing decisions: {n_differ})")
        print(f"  A1 Admitted: {n_admit_a1} | A9 Admitted: {n_admit_a9}")
        print(f"  A9 Auto-Accepted Confident: {policy_a9.telemetry.confident_auto_accepted}")
        print(f"  A9 Auto-Rejected Low Conf: {policy_a9.telemetry.low_conf_auto_rejected}")
        print(f"  A9 Routed to MC: {rec['a9_routed_to_mc']}")
        print(f"  A9 MC Passes: {rec['a9_mc_passes']}")
        print(f"  Parameter Update Norm (A1): {norm_diff_a1:.6f}")
        print(f"  Parameter Update Norm (A9): {norm_diff_a9:.6f}")
        print(f"  Parameter Divergence (A1 vs A9): {param_divergence_a1_a9:.6e}")

    out_json = ROOT / "results" / "a1_a9_equivalence_investigation.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_json}")

if __name__ == "__main__":
    audit_saved_artifacts()
    audit_controller_and_gating_behavior()
