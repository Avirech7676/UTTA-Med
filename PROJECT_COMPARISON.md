# UTTA-Med: Project 1 vs Project 2 Comparison Analysis

**Date:** 2026-10-22  
**Auditor:** Senior AI Research Scientist / Medical Image Analysis Engineer  
**Reference Directory (Project 1):** `C:\Users\avina\OneDrive\Desktop\UTTA-Med`  
**Primary Development Directory (Project 2):** `c:\Users\avina\OneDrive\Desktop\UTTA_MED_V1_FINAL\UTTA_MED_V1_FINAL`

---

## 1. High-Level Comparison Summary

Project 1 served as the early exploratory research repository containing git commit history, legacy notebooks, experimental scratch spaces, and local virtual environment binaries.

Project 2 (`UTTA_MED_V1_FINAL`) is the clean, verified production distribution containing the updated multi-dataset checkpoints, comprehensive test suite, publication-grade manual, and synchronized audit dossiers.

| Comparison Area | Project 1 (`UTTA-Med`) | Project 2 (`UTTA_MED_V1_FINAL`) | Analysis & Recommendation |
| :--- | :--- | :--- | :--- |
| **Primary Status** | Reference / Legacy Sandbox | Active Development & Production | All new developments must be implemented and verified in Project 2. |
| **Checkpoints** | 5 ResNet-18 Camelyon17 models + 3 Faster R-CNN models | 9 ResNet-18 models (Camelyon17, MIDOG, NCT-CRC, PathMNIST, RandStainNA) + 3 Faster R-CNN models | Project 2 has the complete multi-domain evaluation model suite. |
| **Checkpoints Integrity** | SHA-256 verified | Bit-for-bit identical hashes on shared models | Zero divergence in certified baseline weights. |
| **Manuscript Draft** | Draft LaTeX & PDF (5.5 MB) | IEEE Tectonic-compiled 8.0-page certified PDF (308 KB) | Project 2 strictly meets IEEE format and page budget constraints. |
| **Technical Documentation** | Fragmented Markdown notes | 14-Chapter R&D Manual (`.md` & `.docx`), UROP Monograph, Post-Verification Report | Project 2 contains publication and regulatory defense deliverables. |
| **Audit Dossiers** | Present in root | Synchronized into `docs/` and verified | H4 Protocol, H5 Scientific Claims, and Supplementary Reproducibility reports preserved. |
| **Storage Footprint** | ~1.92 GB (includes 916 MB `.venv` and 49 MB `tectonic.exe`) | Clean modular workspace (All standalone zips strictly $< 500\text{ MB}$) | Project 2 is optimized for reliable cloud sync and distribution. |
| **Automated Test Suite** | 114 tests | 120 passing tests (100% pass rate in 18.34s) | Project 2 includes extra fail-closed and immutability regression tests. |

---

## 2. Checkpoint Comparison

| Checkpoint Name | Project 1 Size & Hash | Project 2 Size & Hash | Parity Status |
| :--- | :--- | :--- | :---: |
| `camelyon17_resnet18_source_s42_BEST.pt` | 43.72 MB (`f0b85ec1a89707a8...`) | 43.72 MB (`f0b85ec1a89707a8...`) | **MATCH** |
| `camelyon17_resnet18_source_s7_BEST.pt` | 43.72 MB (`9c18d65ab0a97527...`) | 43.72 MB (`9c18d65ab0a97527...`) | **MATCH** |
| `camelyon17_resnet18_source_s99_BEST.pt` | 43.72 MB (`0b299932abd18419...`) | 43.72 MB (`0b299932abd18419...`) | **MATCH** |
| `camelyon17_resnet18_source_s123_BEST.pt` | 43.72 MB (`cb57d805869f1176...`) | 43.72 MB (`cb57d805869f1176...`) | **MATCH** |
| `camelyon17_resnet18_source_s2024_BEST.pt` | 43.72 MB (`d9bed5d9bc05edd7...`) | 43.72 MB (`d9bed5d9bc05edd7...`) | **MATCH** |
| `source_fasterrcnn_s42.pt` | 158.05 MB (`af9c34437e37cb13...`) | 158.05 MB (`af9c34437e37cb13...`) | **MATCH** |
| `source_fasterrcnn_s7.pt` | 158.05 MB (`3f1ed8e0e3997eac...`) | 158.05 MB (`3f1ed8e0e3997eac...`) | **MATCH** |
| `source_fasterrcnn_s123.pt` | 158.05 MB (`44584d70480a7913...`) | 158.05 MB (`44584d70480a7913...`) | **MATCH** |
| `midog_resnet18_s42_BEST.pt` | Missing | 42.71 MB (`409a9c7d3b9d1052...`) | **Project 2 Only** |
| `nctcrc_resnet18_s42_BEST.pt` | Missing | 42.73 MB (`92631223343f5a77...`) | **Project 2 Only** |
| `pathmnist_r18_s42.pt` | Missing | 42.73 MB (`ce2cfcf0db4378f5...`) | **Project 2 Only** |
| `source_centroids_s42.pt` | Missing | 0.02 MB (`08340599c894de6a...`) | **Project 2 Only** |
| `source_resnet18_randstainna_s42.pt` | Missing | 43.72 MB (`ed2db11c1357b884...`) | **Project 2 Only** |

---

## 3. Structural File Parity

### Key Files Present Exclusively in Project 2:
1. `RESEARCH_AND_DEVELOPMENT_MANUAL.md` & `.docx` (14-Chapter Exhaustive Manual)
2. `UROP_RESEARCH_DRAFT_UTTA_MED.md` & `.docx` (Undergraduate Research Monograph)
3. `POST_VERIFICATION_IMPLEMENTATION_REPORT.md` & `.docx` (Audit and Gap-Closure Report)
4. `PUBLICATION_METADATA_AND_ABSTRACT.md`
5. `UTTA_MED_COMPLETE_UNDER_500MB.zip` (494.69 MB portable standalone package)
6. `UTTA_MED_SUPPLEMENTARY_DETECTION_MODELS.zip` (440.14 MB detector package)
7. `UTTA_MED_SUPPLEMENTARY_VISUAL_OVERLAYS.zip` (145.13 MB visual diagnostics package)

### Key Files Synchronized from Project 1 into Project 2 `docs/`:
1. `docs/H4_FINAL_PROVENANCE_AND_PROTOCOL_DEVIATION_AUDIT.md`
2. `docs/H5_MANUSCRIPT_SCIENTIFIC_CLAIMS_AUDIT.md`
3. `docs/H5_SUPPLEMENTARY_REPRODUCIBILITY_REPORT.md`

---

## 4. Conclusion & Development Plan

Project 2 is the superior, bit-exact, and fully functional codebase. Project 1 will be retained strictly as an archival reference. All new implementations—including Adaptive Uncertainty Estimation, Multi-Signal Gating, Bio-SafeAdapt, and Kaggle GPU automation—are being developed directly in Project 2.
