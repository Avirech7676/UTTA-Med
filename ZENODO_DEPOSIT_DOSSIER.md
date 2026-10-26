# Zenodo Permanent DOI Deposition Dossier: SafeAdapt

**Protocol Stage:** P3 — Post-Submission / Post-Acceptance Archival Protocol  
**Repository Name:** SafeAdapt Research Workspace  
**GitHub Repository:** [https://github.com/Avirech7676/SafeAdapt](https://github.com/Avirech7676/SafeAdapt)  
**Corresponding Authors:** Avinash Reddy Cheerapareddy & Gunnam Sanjana  
**Affiliation:** Department of Computer Science and Engineering, SRM University-AP, Amaravati, Andhra Pradesh 522240, India  
**Target Repository:** Zenodo (CERN / OpenAIRE Open Access Repository)  
**Permanent Target DOI Schema:** `10.5281/zenodo.<RECORD_ID>`  
**Primary Archive Package:** `safeadapt_reproducibility_package.tar.gz`  


---

## 1. Executive Summary & Archival Mandate

In compliance with IEEE and international open-science publication integrity standards, this dossier provides the complete, pre-configured archival manifest for permanently depositing all SafeAdapt model checkpoints, configuration YAMLs, empirical validation tables, and verification test suites to Zenodo upon formal manuscript acceptance.

This permanent record guarantees that every mathematical bound (Theorem 1), empirical distribution claim ($0.9495$ macro-AUROC, $0.5172$ detection $F_1$, $96.0\%$ AJCC staging concordance), and runtime safety profile ($1.8$\,ms rollback, $35.9$\,s WSI latency) remains permanently verifiable and accessible via a persistent Digital Object Identifier (DOI).

---

## 2. Zenodo API Deposition Metadata (`deposition_metadata.json`)

```json
{
  "metadata": {
    "title": "SafeAdapt: Geometry-Preserving Optimal Transport and Fail-Closed Safeguards for Test-Time Adaptation in Digital Pathology — Models, Configs, and Verification Package",
    "upload_type": "software",
    "publication_type": "software",
    "description": "<p>This archive contains the official source code, certified pre-trained source model checkpoints, adapted weights, experimental configuration files, empirical defense tables, and verification suites for <strong>SafeAdapt</strong>: a mathematically bounded, fail-closed test-time adaptation framework for computational digital pathology.</p><p>Key components include: (1) Entropic Optimal Transport (Wasserstein-2) streaming prototype alignment via Sinkhorn-Knopp iterations; (2) Centroid Margin Regularization (CMR) conditioning detection safety on nuclear cellularity fraction; (3) Provable Lipschitz probability drift bounds (Theorem 1) with sub-frame dual-buffer atomic rollbacks (1.8 ms); and (4) Deterministic per-WSI episodic resets ensuring strict patient diagnostic isolation.</p>",
    "creators": [
      {
        "name": "Cheerapareddy, Avinash Reddy",
        "affiliation": "Department of Computer Science and Engineering, SRM University-AP, Amaravati, India",
        "orcid": "0009-0002-3023-747X"
      },
      {
        "name": "Sanjana, Gunnam",
        "affiliation": "Department of Computer Science and Engineering, SRM University-AP, Amaravati, India"
      }
    ],
    "access_right": "open",
    "license": "CC-BY-4.0",
    "keywords": [
      "Test-Time Adaptation",
      "Digital Pathology",
      "Optimal Transport",
      "Fail-Closed Safeguards",
      "Camelyon17",
      "MIDOG 2021",
      "Cellularity Mitosis Ratio",
      "Uncertainty Quantification",
      "Sinkhorn-Knopp",
      "PACS Integration",
      "DICOMweb",
      "Whole-Slide Imaging"
    ],
    "communities": [
      {"identifier": "digital-pathology"},
      {"identifier": "medical-imaging"}
    ],
    "contributors": [
      {
        "name": "Department of Computer Science and Engineering, SRM University-AP",
        "type": "HostingInstitution"
      }
    ]
  }
}
```

---

## 3. Certified Checkpoint Cryptographic Ledger

All model checkpoints are preserved with bit-level integrity verified via SHA-256 hashes:

| Checkpoint Filename | Architecture | Domain / Benchmark | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- | :---: | :--- |
| `camelyon17_resnet18_source_s42_BEST.pt` | ResNet-18 | Camelyon17 (Seed 42 Source) | 45.84 MB | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` |
| `camelyon17_resnet18_source_s123_BEST.pt` | ResNet-18 | Camelyon17 (Seed 123 Source) | 45.84 MB | `cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224` |
| `camelyon17_resnet18_source_s2024_BEST.pt` | ResNet-18 | Camelyon17 (Seed 2024 Source) | 45.84 MB | `d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261` |
| `camelyon17_resnet18_source_s7_BEST.pt` | ResNet-18 | Camelyon17 (Seed 7 Source) | 45.84 MB | `9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b` |
| `camelyon17_resnet18_source_s99_BEST.pt` | ResNet-18 | Camelyon17 (Seed 99 Source) | 45.84 MB | `0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540` |
| `utta_med_v2_resnet18_adapted_s42.pt` | ResNet-18 | Camelyon17 (SafeAdapt-OT Target) | 45.96 MB | `185cb4c7215ad10a82051c6bde14b22ca9c732463b2111542892b748221a9a1b` |
| `source_centroids_s42.pt` | Prototypes | Feature Space Class Centroids | 0.02 MB | `08340599c894de6ae43a57b40a90b96953934e9a2a3841742cc387c83956c2ec` |
| `midog_resnet18_s42_BEST.pt` | ResNet-18 | MIDOG 2021 Mitosis Baseline | 44.79 MB | `409a9c7d3b9d10523221affc477f5a85660009fabd6359999a5b9b95a07c0287` |
| `source_fasterrcnn_s42.pt` | Faster R-CNN | MIDOG 2021 Mitosis Detector (s42) | 165.73 MB | `af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003` |
| `source_fasterrcnn_s123.pt` | Faster R-CNN | MIDOG 2021 Mitosis Detector (s123) | 165.73 MB | `44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348` |
| `source_fasterrcnn_s7.pt` | Faster R-CNN | MIDOG 2021 Mitosis Detector (s7) | 165.73 MB | `3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e` |
| `nctcrc_resnet18_s42_BEST.pt` | ResNet-18 | NCT-CRC-HE 9-Class Colorectal | 44.80 MB | `92631223343f5a7770b4da7671b9c7f2f26eee052c00807f263cfb4300b33d8e` |
| `pathmnist_r18_s42.pt` | ResNet-18 | MedMNIST PathMNIST Benchmark | 44.80 MB | `ce2cfcf0db4378f56dc3013e079d3095f150e91f0013639bb6c5774655703f73` |

---

## 4. Configuration Registry

All 16 experimental configuration specifications are archived under `configs/`:

| Configuration File | Target Module | SHA-256 Checksum |
| :--- | :--- | :--- |
| `configs/v2/safeadapt_e2_frozen.yaml` | Production Hospital 2 SafeAdapt | `3042225de8be5bd013a5725898a787802714004fb2b3ee95d61ba14e18df1418` |
| `configs/v2/safeadapt_e1_frozen.yaml` | Production Hospital 1 SafeAdapt | `185eb3d4d064ff281114189b33bb0403a9db35717784bfbe46dfd0a41c288145` |
| `configs/v2/safeadapt_c1_monitor.yaml` | Runtime Monitor Safeguards | `4e84bd94d3c734d3edbbe1c3120c11e05685a9d6ea8c5b30e3e408bea8717860` |
| `configs/v2/safeadapt_c2_calibrated.yaml` | Conformal Temperature Calibration | `15a0f59836d4d2f4415458664b3f5af731391bbf7ec3936bc1d957bc5840fa3f` |
| `configs/v2/gating_confidence.yaml` | Confidence Gating Policy | `68ea8d5e5779bfdcb1e043170e21c1045b0b978a9b5f8eee7c7eaa5c82cf780c` |
| `configs/v2/gating_mc_only.yaml` | MC-Dropout Predictive Variance Gate | `9e5394b43a73a2580876be76bec3505f2a4c911f25df9351f2582417ee645061` |
| `configs/v2/gating_hybrid.yaml` | Conformal Hybrid Gate | `14958fdc10a57421927c41dbf26ab05e6cc769475f1465cc18a1e6b27701a1ef` |
| `configs/v2/gating_frozen_source.yaml` | Static Source Baseline | `a45513152833f07fece77839bec134601e07e39501d93c7d51fe4a848e003c6d` |
| `configs/v2/safeadapt_minimal.yaml` | Low-Footprint Edge Configuration | `8901fc4935bb4eebc778d18699482d1bd09585d68c8e1ec9755edc5d14e84ddb` |
| `configs/camelyon17.yaml` | WILDS Data Ingestion & Preprocessing | `a2362b5db0adfdc0d39f6544f55530563c6a0c6a2be814ccb63f97ce06e07741` |
| `configs/nct_crc_evidential.yaml` | NCT-CRC Multi-Class Evidential Setup | `b7288439e82e2fa8eaad7f5725e193bbc6005ec74f236c4f592ecb3bff074b34` |
| `configs/tau_by_seed.yaml` | Per-Seed Calibrated Gating Thresholds | `79722d4a5554c90b8e4ffec0b4520e98a90ce32cb3b9c07b57a976f22c92c153` |
| `configs/authoritative_resnet18_utta.yaml` | Architecture Specification | `3de29be528704dc8f535f514ff0d88c58ffb75c9027318fb256519bfa520971b` |
| `configs/FINAL/evaluation.yaml` | Evaluation Protocol Definition | `70cb1325f63654fd8c5aad67daf572702a6d3b66e94b1379cbcc11629cab8d9f` |
| `configs/FINAL/source_resnet18.yaml` | Source Training Hyperparameters | `9c9e07ce29f15f9ee3486f47338ed9adb576732d0e50dc4ebe9d153be8f54109` |
| `configs/FINAL/tta_utta_med.yaml` | Adaptation Hyperparameters | `fabb187ff6287a9805a0f9f5d0153ad7473714355db83dc25dfb3e966d41188d` |

---

## 5. Automated Deposition Procedure (Step-by-Step)

Upon formal acceptance notification, execute the following three steps to publish the record and mint the persistent DOI:

### Step 1: Generate Package and Validate Cryptographic Manifest
```bash
python scripts/prepare_zenodo_bundle.py
python scripts/verify_manuscript_tables.py
pytest tests/
```

### Step 2: Create Zenodo Deposition Draft via REST API
```bash
# Export your Zenodo Personal Access Token
export ZENODO_TOKEN="<YOUR_ZENODO_ACCESS_TOKEN>"

# Create deposition container
DEPOSITION_ID=$(curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -X POST "https://zenodo.org/api/deposit/depositions" \
  -d @deposition_metadata.json | jq -r '.id')

echo "Zenodo Deposition ID: $DEPOSITION_ID"
```

### Step 3: Upload Artifacts and Checkpoints
```bash
# Upload tarball package
curl -H "Authorization: Bearer $ZENODO_TOKEN" \
  -F "file=@safeadapt_reproducibility_package.tar.gz" \
  "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID/files"

# Upload manuscript and supplement PDFs
curl -H "Authorization: Bearer $ZENODO_TOKEN" \
  -F "file=@paper/ieee/utta_med_camera_ready.pdf" \
  "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID/files"

curl -H "Authorization: Bearer $ZENODO_TOKEN" \
  -F "file=@paper/ieee/supplement.pdf" \
  "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID/files"

# Publish the deposition to mint permanent DOI
curl -H "Authorization: Bearer $ZENODO_TOKEN" \
  -X POST "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID/actions/publish"
```

Once published, Zenodo issues a persistent DOI of the form `10.5281/zenodo.<DEPOSITION_ID>`, which is then inserted into the camera-ready manuscript Data and Code Availability statement.
