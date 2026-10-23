"""Unit tests for Attention MIL and Clinical WSI Staging."""
import numpy as np
import torch
from src.models.attn_mil import GatedAttentionMIL, TestTimeAttentionAdapter
from src.evaluation.wsi_staging import classify_ajcc_ptnm, compute_metastasis_diameter_mm, evaluate_slide_staging


def test_gated_attention_mil_forward():
    mil = GatedAttentionMIL(in_features=64, hidden_dim=32, num_classes=2)
    # 50 patches from one slide
    h = torch.randn(50, 64)
    logits, attn = mil(h)
    assert logits.shape == (1, 2)
    assert attn.shape == (50, 1)
    # Attention must sum to 1.0
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5)


def test_test_time_attention_adapter():
    mil = GatedAttentionMIL(in_features=64, hidden_dim=32, num_classes=2)
    adapter = TestTimeAttentionAdapter(mil, lr=1e-4)
    h = torch.randn(40, 64)
    res = adapter.adapt_slide(h)
    assert res["slide_logits"].shape == (1, 2)
    assert res["slide_probs"].shape == (1, 2)
    assert res["attention"].shape == (40, 1)
    assert "loss" in res


def test_wsi_staging_ajcc():
    # Negative slide
    assert classify_ajcc_ptnm(0.0) == "pN0"
    # ITC
    assert classify_ajcc_ptnm(0.15) == "pN0(i+)"
    # Micro-metastasis
    assert classify_ajcc_ptnm(1.2) == "pN1mi"
    # Macro-metastasis
    assert classify_ajcc_ptnm(3.5) == "pN1a"


def test_diameter_computation():
    coords = [(0, 0), (96, 96)]
    d_mm = compute_metastasis_diameter_mm(coords, patch_size_px=96, mpp=0.5)
    assert d_mm > 0.0
    # Evaluate full slide dictionary
    probs = np.array([0.9, 0.8, 0.1, 0.05])
    patch_coords = [(0, 0), (100, 100), (500, 500), (600, 600)]
    res = evaluate_slide_staging(probs, patch_coords, true_tumor_fraction=0.50)
    assert res["pred_stage"] in ["pN0", "pN0(i+)", "pN1mi", "pN1a"]
    assert "burden_absolute_error" in res
