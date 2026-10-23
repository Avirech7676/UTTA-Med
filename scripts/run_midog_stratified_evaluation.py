#!/usr/bin/env python3
"""Stage H3.2-R3c: Runner for Stratified Full-Canvas Target Adaptation across MIDOG Scanners.

Executes the upgraded target adaptation protocol with:
1. Stratified Full-Canvas Grid Sampling covering entire WSI dimensions ([0, W] x [0, H]).
2. Multi-Scanner target selection (Scanner 1, 2, 3, or 4) for Leave-One-Scanner-Out evaluation.
3. Formal case-clustered bootstrap contrasts (B=1000).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Stratified Multi-Scanner Adaptation")
    parser.add_argument("--target_scanner", type=int, default=3, choices=[1, 2, 3, 4],
                        help="Target scanner ID (1=3DHISTECH, 2=Hamamatsu XR, 3=Aperio CS2, 4=Hamamatsu S360)")
    parser.add_argument("--patches_per_case", type=int, default=22,
                        help="Annotation-independent tissue crops per case (22 yields ~1100 tiles)")
    parser.add_argument("--output_dir", type=str, default="results/generalization/h3/h3_2_r3c_stratified")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2-case smoke test")
    args = parser.parse_args()

    script_path = Path("notebooks/official/kaggle_midog_target_adaptation_h3_2_r3a.py")
    if not script_path.exists():
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(script_path),
        "--target_scanner", str(args.target_scanner),
        "--target_patches_per_case", str(args.patches_per_case),
        "--output_dir", str(args.output_dir),
    ]
    if args.smoke_test:
        cmd.append("--smoke_test")

    print("=" * 80)
    print(f"[*] Launching Stratified Canvas-Wide Target Adaptation (Scanner {args.target_scanner})")
    print(f"[*] Command: {' '.join(cmd)}")
    print("=" * 80)

    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
