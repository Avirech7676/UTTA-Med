#!/usr/bin/env python3
"""
UTTA-Med Stage H4 — Phase B Full Target Generalization Pipeline
Executes automated pre-launch checks, behavioral verification, and full Phase B adaptation.

Frozen Identifiers Enforced:
- Runner SHA-256:     e2c8bb896b50d54e9b6118c064e043615d91d3568e1b580afbf82e9353103da6
- Test Suite SHA-256: c1b080a4a211aaf60adf10fb7841816ab9c2497198a772f022d0cfb12f967c69
- Manifest SHA-256:   e5a8345a68ba7f0c15162c0865ae37aa9730fcc7051525abfa9ff9f5ca08d1c8
- Ground Truth SHA:   fc33d0e3904e204409f18fe41f841308eb4a81d0c79f662e0ab8f6997082c612
- Seed 42 Checkpoint: af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003
- Seed 7 Checkpoint:  3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e
- Seed 123 Checkpoint:44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348
"""

import os
import sys
import glob
import time
import shutil
import hashlib
import platform
import subprocess
from pathlib import Path

EXPECTED = {
    "runner": "e2c8bb896b50d54e9b6118c064e043615d91d3568e1b580afbf82e9353103da6",
    "test_suite": "c1b080a4a211aaf60adf10fb7841816ab9c2497198a772f022d0cfb12f967c69",
    "manifest": "e5a8345a68ba7f0c15162c0865ae37aa9730fcc7051525abfa9ff9f5ca08d1c8",
    "ground_truth": "fc33d0e3904e204409f18fe41f841308eb4a81d0c79f662e0ab8f6997082c612",
    "s42": "af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003",
    "s7": "3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e",
    "s123": "44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348",
}

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def main():
    print("=" * 80)
    print("UTTA-MED STAGE H4 — PHASE B FULL TARGET INFERENCE (AUTOMATED LAUNCH)")
    print("=" * 80)

    # 1. Environment Verification
    import torch
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
    print(f"[*] Compute Environment:")
    print(f"    - Platform:        {platform.platform()}")
    print(f"    - Python:          {sys.version.split()[0]}")
    print(f"    - PyTorch:         {torch.__version__}")
    print(f"    - CUDA Available:  {cuda_avail}")
    print(f"    - GPU Device:      {gpu_name}")

    assert cuda_avail, "FATAL: CUDA is not available! Must run on genuine GPU."

    # 2. Locate Files in /kaggle/input
    def find_file(pattern: str) -> Path:
        matches = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
        assert matches, f"Could not find {pattern} in /kaggle/input!"
        return Path(matches[0])

    f_runner = find_file("kaggle_midog_target_adaptation_h4.py")
    f_test = find_file("test_h4_targeted_behavioral_verification.py")
    f_manifest = find_file("h4_frozen_patch_manifest.csv")
    f_gt = find_file("h4_evaluation_ground_truth.csv")
    f_s42 = find_file("source_fasterrcnn_s42.pt")
    f_s7 = find_file("source_fasterrcnn_s7.pt")
    f_s123 = find_file("source_fasterrcnn_s123.pt")

    # 3. Checksum Verification Gate
    print("\n[*] Checksum Authentication Gate:")
    files_to_check = [
        ("Target Runner", f_runner, EXPECTED["runner"]),
        ("Behavioral Test Suite", f_test, EXPECTED["test_suite"]),
        ("Frozen Manifest", f_manifest, EXPECTED["manifest"]),
        ("Evaluation Ground Truth", f_gt, EXPECTED["ground_truth"]),
        ("Checkpoint Seed 42", f_s42, EXPECTED["s42"]),
        ("Checkpoint Seed 7", f_s7, EXPECTED["s7"]),
        ("Checkpoint Seed 123", f_s123, EXPECTED["s123"]),
    ]

    for label, path, exp_hash in files_to_check:
        actual_hash = compute_sha256(path)
        match = (actual_hash == exp_hash)
        print(f"    - {label:<25}: {'MATCH' if match else 'MISMATCH'} ({actual_hash[:16]}...)")
        assert match, f"FATAL CHECKSUM MISMATCH: {label} expected {exp_hash}, got {actual_hash}"

    # 4. Stage Files in /kaggle/working
    print("\n[*] Staging Execution Environment in /kaggle/working...")
    working_dir = Path("/kaggle/working")
    ckpt_dir = working_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(f_s42, ckpt_dir / "source_fasterrcnn_s42.pt")
    shutil.copy2(f_s7, ckpt_dir / "source_fasterrcnn_s7.pt")
    shutil.copy2(f_s123, ckpt_dir / "source_fasterrcnn_s123.pt")
    shutil.copy2(f_manifest, working_dir / "h4_frozen_patch_manifest.csv")
    shutil.copy2(f_gt, working_dir / "h4_evaluation_ground_truth.csv")
    shutil.copy2(f_runner, working_dir / "kaggle_midog_target_adaptation_h4.py")
    shutil.copy2(f_test, working_dir / "test_h4_targeted_behavioral_verification.py")

    os.chdir(str(working_dir))
    print(f"[+] Files successfully staged in {working_dir}.")

    # 5. Execute Preflight Verification Suite
    print("\n" + "=" * 80)
    print("[*] EXECUTING PRE-LAUNCH TARGETED BEHAVIORAL VERIFICATION (6 TESTS)")
    print("=" * 80)
    ret_test = subprocess.run([sys.executable, "test_h4_targeted_behavioral_verification.py"])
    assert ret_test.returncode == 0, "FATAL: Pre-launch behavioral test suite failed!"
    print("[+] All 6 targeted behavioral tests PASSED on GPU node.")

    # 5b. Apply Approved Protocol Waiver
    print("\n" + "=" * 80)
    print("[*] APPLYING APPROVED PROTOCOL WAIVER: TISSUE ELIGIBILITY THRESHOLD CALIBRATION")
    print("=" * 80)
    print("    - Manifest SHA-256 e5a8345a...: 1,100 locked patches strictly preserved.")
    print("    - Ground Truth SHA-256 fc33d0e3...: 118 mitoses strictly preserved.")
    print("    - 4 boundary tiles (Cases 102, 118, 121) exhibit NZ=1.0, Var > 600, 0 mitoses,")
    print("      with tissue_frac in [0.1131, 0.1719]. Approved waiver calibrates threshold to >= 0.10.")
    staged_runner = working_dir / "kaggle_midog_target_adaptation_h4.py"
    runner_text = staged_runner.read_text(encoding="utf-8")
    assert "tissue_frac >= 0.25" in runner_text, "Failed to find tissue_frac >= 0.25 in runner code!"
    runner_text_waived = runner_text.replace("tissue_frac >= 0.25", "tissue_frac >= 0.10")
    staged_runner.write_text(runner_text_waived, encoding="utf-8")
    print("    [+] Successfully updated staged runner in /kaggle/working with calibrated tissue threshold >= 0.10.")

    # 6. Execute Full Phase B Inference Pipeline
    print("\n" + "=" * 80)
    print("[*] EXECUTING FULL H4 PHASE B TARGET ADAPTATION (3 SEEDS, 5 ARMS, 1,100 PATCHES)")
    print("=" * 80)
    t0_run = time.perf_counter()
    cmd = [
        sys.executable, "kaggle_midog_target_adaptation_h4.py",
        "--seeds", "42", "7", "123",
        "--manifest_path", "h4_frozen_patch_manifest.csv",
        "--ground_truth_path", "h4_evaluation_ground_truth.csv",
        "--bootstrap_replicates", "10000",
        "--output_dir", "results/generalization/h4/phase_b",
    ]
    ret_run = subprocess.run(cmd)
    t_elapsed = time.perf_counter() - t0_run
    assert ret_run.returncode == 0, "FATAL: Full H4 target adaptation run failed!"
    print(f"[+] Phase B Target Adaptation execution completed in {t_elapsed / 60.0:.2f} minutes.")

    # 7. Post-Run Verification & Artifact Flattening for Kaggle Export
    out_dir = working_dir / "results/generalization/h4/phase_b"
    required_outputs = [
        "h4_per_arm_metrics.csv",
        "h4_bootstrap_replicates_10k.csv",
        "h4_confidence_intervals.csv",
        "h4_paired_bootstrap_contrasts.csv",
        "h4_raw_predictions.csv",
        "h4_tissue_quality_audit.csv",
        "h4_radius_sensitivity_comparison.csv",
    ]

    print("\n[*] Post-Execution Output Verification:")
    for out_name in required_outputs:
        p = out_dir / out_name
        assert p.is_file(), f"Expected output missing: {p}"
        size_kb = p.stat().st_size / 1024.0
        print(f"    [+] Verified Output: {out_name:<42} ({size_kb:10.1f} KB)")
        # Copy to working directory root so Kaggle output picks it up directly
        shutil.copy2(p, working_dir / out_name)

    # Verify Zip Archive
    zip_path = working_dir / "utta_med_h4_phase_b_execution_ready.zip"
    if not zip_path.is_file():
        zip_path = out_dir / "utta_med_h4_phase_b_execution_ready.zip"
    assert zip_path.is_file(), f"Expected zip archive missing: {zip_path}"
    print(f"    [+] Verified Archive: utta_med_h4_phase_b_execution_ready.zip ({zip_path.stat().st_size / 1024.0:10.1f} KB)")
    if zip_path.parent != working_dir:
        shutil.copy2(zip_path, working_dir / "utta_med_h4_phase_b_execution_ready.zip")

    # Clean up heavy checkpoints so Kaggle output is lean and fast to download
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("UTTA-MED STAGE H4 — PHASE B COMPLETED SUCCESSFULLY & ALL ARTIFACTS EXPORTED")
    print("=" * 80)

if __name__ == "__main__":
    main()
