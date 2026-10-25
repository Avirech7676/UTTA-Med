# Day 1: Reproducibility Verification Report

**Generated:** Windows-11-10.0.26300-SP0 | Python 3.14.6 | PyTorch 2.14.0+cpu

## 1. Verified Headline Results (5 Seeds)

| Method | Macro AUROC | F1-Score | Error Rate |
| :--- | :---: | :---: | :---: |
| **source** | 0.9297 ± 0.0092 | 0.8069 ± 0.0247 | 0.1660 ± 0.0156 |
| **tent** | 0.6981 ± 0.3237 | 0.5057 ± 0.4111 | 0.2929 ± 0.1779 |
| **eata** | 0.9215 ± 0.0263 | 0.8096 ± 0.0316 | 0.1606 ± 0.0218 |
| **random** | 0.6967 ± 0.3044 | 0.4870 ± 0.3850 | 0.3069 ± 0.1647 |
| **confidence** | 0.9355 ± 0.0218 | 0.8310 ± 0.0321 | 0.1455 ± 0.0227 |
| **entropy** | 0.9355 ± 0.0218 | 0.8310 ± 0.0321 | 0.1455 ± 0.0227 |
| **utta** | 0.9351 ± 0.0218 | 0.8296 ± 0.0325 | 0.1465 ± 0.0230 |

## 2. Checkpoint Provenance Hashes

| Checkpoint File | SHA256 Hash |
| :--- | :--- |
| `camelyon17_resnet18_source_s123_BEST.pt` | `cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224` |
| `camelyon17_resnet18_source_s2024_BEST.pt` | `d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261` |
| `camelyon17_resnet18_source_s42_BEST.pt` | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` |
| `camelyon17_resnet18_source_s7_BEST.pt` | `9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b` |
| `camelyon17_resnet18_source_s99_BEST.pt` | `0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540` |
| `midog_resnet18_s42_BEST.pt` | `409a9c7d3b9d10523221affc477f5a85660009fabd6359999a5b9b95a07c0287` |
| `nctcrc_resnet18_s42_BEST.pt` | `92631223343f5a7770b4da7671b9c7f2f26eee052c00807f263cfb4300b33d8e` |
| `pathmnist_r18_s42.pt` | `ce2cfcf0db4378f56dc3013e079d3095f150e91f0013639bb6c5774655703f73` |
| `source_centroids_s42.pt` | `08340599c894de6ae43a57b40a90b96953934e9a2a3841742cc387c83956c2ec` |
| `source_fasterrcnn_s123.pt` | `44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348` |
| `source_fasterrcnn_s42.pt` | `af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003` |
| `source_fasterrcnn_s7.pt` | `3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e` |
