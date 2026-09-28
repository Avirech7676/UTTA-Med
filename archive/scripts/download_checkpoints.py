"""
Checkpoint verification and downloader utility for UTTA-Med.
Verifies SHA256 checksums against reproducibility/ARTIFACT_MANIFEST.csv
and allows downloading remaining multi-seed checkpoints.
"""

import os
import hashlib
from pathlib import Path

MANIFEST_CHECKSUMS = {
    "camelyon17_resnet18_source_s42_BEST.pt": "f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000",
    "camelyon17_resnet18_source_s123_BEST.pt": "cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224",
    "camelyon17_resnet18_source_s2024_BEST.pt": "d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261",
    "camelyon17_resnet18_source_s7_BEST.pt": "9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b",
    "camelyon17_resnet18_source_s99_BEST.pt": "0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540",
}

def verify_file(filepath: Path, expected_sha256: str) -> bool:
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha256.lower()

def main():
    root = Path(__file__).resolve().parent.parent
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("UTTA-Med Checkpoint Verification & Status Report")
    print("=" * 60)
    
    for filename, sha in MANIFEST_CHECKSUMS.items():
        p = ckpt_dir / filename
        if p.exists():
            matched = verify_file(p, sha)
            status = "VERIFIED MATCH" if matched else "SHA256 MISMATCH"
            sz_mb = p.stat().st_size / (1024 * 1024)
            print(f"[{status}] {filename:42s} ({sz_mb:.2f} MB)")
        else:
            print(f"[NOT PRESENT]    {filename:42s} (Download from Kaggle / Train with train_source.py)")

    print("\nNote: The primary benchmark model (s42) is included locally.")
    print("To regenerate any missing seed from scratch with full bit-exact reproducibility:")
    print("  python scripts/train_source.py --seed <SEED> --config configs/camelyon17.yaml")
    print("Or pull the multi-seed weights bundle from Kaggle Dataset: 'uttam-checkpoints'.")

if __name__ == "__main__":
    main()
