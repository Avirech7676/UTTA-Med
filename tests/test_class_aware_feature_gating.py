"""
Unit Tests for Class-Aware Feature-Distance Gating Policy
Tests:
  1. Feature dimension verification & error handling (no representation truncation)
  2. Threshold behavior and admission decisions (margin agreement)
  3. Deterministic inference across multiple runs
  4. PyTorch and NumPy interoperability
  5. Distance metric behavior (cosine vs normalized euclidean)
"""

import pytest
import numpy as np
import torch
from src.v2.gating.class_aware_feature_gating import ClassAwareFeatureGatingPolicy

def test_feature_dimension_verification():
    policy = ClassAwareFeatureGatingPolicy(embed_dim=512, num_classes=2)
    
    # Valid dimensions
    feat_valid = np.random.randn(4, 512).astype(np.float32)
    cent_valid = np.random.randn(2, 512).astype(np.float32)
    logits_valid = np.random.randn(4, 2).astype(np.float32)
    
    res = policy.evaluate(logits_valid, feat_valid, cent_valid)
    assert res["admit_mask"].shape == (4,)
    assert res["joint_scores"].shape == (4,)
    
    # Incompatible feature dimension (e.g., 256)
    feat_invalid = np.random.randn(4, 256).astype(np.float32)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        policy.evaluate(logits_valid, feat_invalid, cent_valid)

    # Incompatible centroid dimension
    cent_invalid = np.random.randn(2, 128).astype(np.float32)
    with pytest.raises(ValueError, match="Centroid dimension mismatch"):
        policy.evaluate(logits_valid, feat_valid, cent_invalid)

def test_threshold_behavior_and_admission():
    policy = ClassAwareFeatureGatingPolicy(
        w_conf=0.5,
        w_feat=0.5,
        tau_joint=0.75,
        tau_margin=0.20,
        enforce_margin_agreement=True,
        embed_dim=512,
        num_classes=2
    )

    # Synthetic setup:
    # Centroid 0: [1, 0, 0, ...]
    # Centroid 1: [0, 1, 0, ...]
    c0 = np.zeros((1, 512), dtype=np.float32)
    c0[0, 0] = 1.0
    c1 = np.zeros((1, 512), dtype=np.float32)
    c1[0, 1] = 1.0
    centroids = np.vstack([c0, c1])

    # Sample A: High confidence in class 0, and feature exactly aligns with c0
    logits_a = np.array([[5.0, -5.0]], dtype=np.float32) # p_0 ~ 1.0
    feat_a = c0.copy()

    res_a = policy.evaluate(logits_a, feat_a, centroids)
    assert res_a["pred_classes"][0] == 0
    assert res_a["head_manifold_agreement"][0] == True
    assert res_a["margins"][0] > 0.0 # d_alt=1.0, d_pred=0.0 -> margin=+1.0
    assert res_a["admit_mask"][0] == True
    assert res_a["joint_scores"][0] > 0.85

    # Sample B: High confidence in class 0, but feature aligns with c1 (CONTRADICTION!)
    logits_b = np.array([[5.0, -5.0]], dtype=np.float32) # head predicts class 0
    feat_b = c1.copy() # feature is at centroid 1!

    res_b = policy.evaluate(logits_b, feat_b, centroids)
    assert res_b["pred_classes"][0] == 0
    assert res_b["nearest_centroid_classes"][0] == 1
    assert res_b["margins"][0] < 0.0 # d_alt(c1)=0.0, d_pred(c0)=1.0 -> margin=-1.0
    assert res_b["head_manifold_agreement"][0] == False
    # CRITICAL: Even though confidence is near 100%, admission MUST be rejected!
    assert res_b["admit_mask"][0] == False

def test_deterministic_inference():
    policy = ClassAwareFeatureGatingPolicy(embed_dim=512, num_classes=2)
    np.random.seed(42)
    logits = np.random.randn(10, 2).astype(np.float32)
    features = np.random.randn(10, 512).astype(np.float32)
    centroids = np.random.randn(2, 512).astype(np.float32)

    res1 = policy.evaluate(logits, features, centroids)
    res2 = policy.evaluate(logits, features, centroids)

    np.testing.assert_array_equal(res1["admit_mask"], res2["admit_mask"])
    np.testing.assert_allclose(res1["joint_scores"], res2["joint_scores"], atol=1e-7)
    np.testing.assert_allclose(res1["margins"], res2["margins"], atol=1e-7)

def test_pytorch_and_numpy_interop():
    policy = ClassAwareFeatureGatingPolicy(embed_dim=512, num_classes=2)
    torch.manual_seed(42)
    logits_t = torch.randn(8, 2)
    features_t = torch.randn(8, 512)
    centroids_t = torch.randn(2, 512)

    res_t = policy.evaluate(logits_t, features_t, centroids_t)
    res_np = policy.evaluate(logits_t.numpy(), features_t.numpy(), centroids_t.numpy())

    np.testing.assert_array_equal(res_t["admit_mask"], res_np["admit_mask"])
    np.testing.assert_allclose(res_t["joint_scores"], res_np["joint_scores"], atol=1e-6)
    np.testing.assert_allclose(res_t["margins"], res_np["margins"], atol=1e-6)

def test_distance_metrics():
    policy_cos = ClassAwareFeatureGatingPolicy(distance_metric="cosine", embed_dim=512)
    policy_euc = ClassAwareFeatureGatingPolicy(distance_metric="normalized_euclidean", embed_dim=512)

    feat = np.random.randn(4, 512).astype(np.float32)
    cent = np.random.randn(2, 512).astype(np.float32)

    d_cos, _ = policy_cos.compute_distances(feat, cent)
    d_euc, _ = policy_euc.compute_distances(feat, cent)

    assert d_cos.shape == (4, 2)
    assert d_euc.shape == (4, 2)
    assert np.all(d_cos >= 0.0)
    assert np.all(d_euc >= 0.0)
