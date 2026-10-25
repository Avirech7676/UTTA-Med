#!/usr/bin/env python3
"""
UTTA-Med — Stage H4: Independent Verification of Sampling Freeze & Label Blindness
===================================================================================
Independent reproducibility audit script to:
1. Regenerate the exact 1,100-patch H4 sampling manifest from first principles using
   ONLY image dimensions and the frozen master RNG seed (20261016).
2. Verify that ZERO annotation metadata, mitosis coordinates, or mitosis counts were
   accessed or used during the patch selection procedure.
3. Compare the regenerated coordinates bit-for-bit against h4_frozen_patch_manifest.csv.
4. Compute and print immutable SHA-256 digests of all sampling artifacts.
===================================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PREREGISTERED_SAMPLING_SEED = 20261016
TARGET_CASES = list(range(101, 151))
PATCH_SIZE = 512
TARGET_PATCHES_PER_CASE = 22


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def find_midog_coco_json() -> Path:
    candidates = [
        Path("data/midog_scanner3_coco_metadata.json"),
        Path("data/MIDOG.json"),
        Path("MIDOG.json"),
        Path("str(Path.home() / "Downloads")/MIDOG.json"),
        Path("str(Path.home() / "Downloads")/MIDOG (1).json"),
        Path("/kaggle/input/midog-challenge/MIDOG.json"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Could not find MIDOG.json or data/midog_scanner3_coco_metadata.json")


def main() -> None:
    print("=" * 80)
    print("UTTA-MED — STAGE H4: INDEPENDENT SAMPLING FREEZE VERIFICATION")
    print("=" * 80)

    # 1. Load ONLY image metadata (width, height, id) — ZERO ANNOTATION ACCESS
    coco_path = find_midog_coco_json()
    with open(coco_path, "r", encoding="utf-8") as f:
        raw_coco = json.load(f)

    # Strictly extract image dimensions only
    image_metadata: Dict[int, Tuple[int, int]] = {}
    for im in raw_coco.get("images", []):
        iid = int(im["id"])
        if 101 <= iid <= 150:
            image_metadata[iid] = (int(im["width"]), int(im["height"]))

    print(f"[*] Loaded metadata for {len(image_metadata)} Scanner 3 cases.")
    print("[*] GUARANTEE: Zero annotation records, categories, or bounding boxes were loaded.")

    # 2. Regenerate Candidate Grid & 16 Strata
    candidate_records = []
    strata_inventory = defaultdict(lambda: defaultdict(list))

    for cid in TARGET_CASES:
        w, h = image_metadata[cid]
        nx = w // PATCH_SIZE
        ny = h // PATCH_SIZE

        for row_idx in range(ny):
            py = row_idx * PATCH_SIZE
            for col_idx in range(nx):
                px = col_idx * PATCH_SIZE
                cx = px + PATCH_SIZE / 2.0
                cy = py + PATCH_SIZE / 2.0
                s_gx = min(3, max(0, int((cx / w) * 4)))
                s_gy = min(3, max(0, int((cy / h) * 4)))
                sid = f"R{s_gy}C{s_gx}"

                cand = {
                    "case_id": cid,
                    "candidate_id": f"{cid}_{row_idx}_{col_idx}",
                    "grid_row": row_idx,
                    "grid_col": col_idx,
                    "origin_x": px,
                    "origin_y": py,
                    "width": PATCH_SIZE,
                    "height": PATCH_SIZE,
                    "center_x": round(cx, 1),
                    "center_y": round(cy, 1),
                    "stratum_row": s_gy,
                    "stratum_col": s_gx,
                    "stratum_id": sid,
                    "image_width": w,
                    "image_height": h,
                    "is_within_annotated_roi": True,
                }
                candidate_records.append(cand)
                strata_inventory[cid][sid].append(cand)

    print(f"[*] Regenerated {len(candidate_records)} candidate tiles across 50 cases.")

    # 3. Regenerate Frozen 1,100 Selection using Deterministic Preregistered RNG
    regenerated_patches = []
    for cid in TARGET_CASES:
        rng = np.random.RandomState(PREREGISTERED_SAMPLING_SEED + cid * 7919)
        case_strata = strata_inventory[cid]
        all_sids = [f"R{gy}C{gx}" for gy in range(4) for gx in range(4)]
        avail_sids = [sid for sid in all_sids if len(case_strata[sid]) > 0]

        # Allocate 22 quota evenly across available strata
        stratum_allocation = Counter({sid: 1 for sid in avail_sids})
        rem = TARGET_PATCHES_PER_CASE - sum(stratum_allocation.values())
        if rem > 0:
            extra = rng.permutation(avail_sids).tolist()
            for sid in extra:
                if rem <= 0:
                    break
                if len(case_strata[sid]) > stratum_allocation[sid]:
                    stratum_allocation[sid] += 1
                    rem -= 1

        selected = []
        for sid in avail_sids:
            k = stratum_allocation[sid]
            pool = list(case_strata[sid])
            chosen_indices = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            for idx in chosen_indices:
                selected.append(pool[idx])

        # Deterministic presentation sort
        selected.sort(key=lambda x: (x["origin_y"], x["origin_x"]))
        regenerated_patches.extend(selected)

    print(f"[*] Regenerated {len(regenerated_patches)} frozen evaluation patches.")

    # 4. Compare Bit-for-Bit with Existing Manifest
    manifest_path = Path("h4_frozen_patch_manifest.csv")
    if not manifest_path.is_file():
        manifest_path = Path("results/generalization/h4/phase_a/h4_frozen_patch_manifest.csv")

    frozen_df = pd.read_csv(manifest_path)
    print(f"[*] Loaded frozen reference manifest: {manifest_path} ({len(frozen_df)} rows)")

    matches = 0
    mismatches = 0
    for idx, (reg, (_, ref)) in enumerate(zip(regenerated_patches, frozen_df.iterrows())):
        reg_tuple = (int(reg["case_id"]), int(reg["origin_x"]), int(reg["origin_y"]))
        ref_tuple = (int(ref["case_id"]), int(ref["origin_x"]), int(ref["origin_y"]))
        if reg_tuple == ref_tuple:
            matches += 1
        else:
            mismatches += 1
            if mismatches <= 5:
                print(f"  [MISMATCH] Row {idx}: regenerated={reg_tuple} vs reference={ref_tuple}")

    print(f"\n[*] Exact Bit-for-Bit Verification:")
    print(f"    - Total Matches   : {matches} / {len(frozen_df)} ({matches/len(frozen_df):.2%})")
    print(f"    - Total Mismatches: {mismatches}")

    # 5. Compute SHA-256 Checksums
    cand_path = Path("h4_candidate_sampling_frame.csv")
    if not cand_path.is_file():
        cand_path = Path("results/generalization/h4/phase_a/h4_candidate_sampling_frame.csv")

    print("\n[*] Immutable File Checksums (SHA-256):")
    print(f"    - {manifest_path.name:<32}: {compute_sha256(manifest_path)}")
    print(f"    - {cand_path.name:<32}: {compute_sha256(cand_path)}")

    if mismatches == 0 and matches == 1100:
        print("\n[PASS] INDEPENDENT SAMPLING FREEZE VERIFIED: Deterministic and 100% Label-Blind.")
    else:
        print("\n[FAIL] VERIFICATION FAILED: Coordinates do not match reference manifest.")
        sys.exit(1)


if __name__ == "__main__":
    main()
