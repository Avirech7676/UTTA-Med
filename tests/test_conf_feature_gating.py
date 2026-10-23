"""Unit tests for ConfidenceFeatureGatingPolicy."""

import pytest
import torch
import torch.nn as nn

from src.models.resnet18 import build_resnet18
from src.v2.gating.confidence_feature_gating import (
    ConfidenceFeatureGatingPolicy,
    ConfFeatureDecision,
)


def test_confidence_feature_gating_initialization():
    policy = ConfidenceFeatureGatingPolicy(w_conf=0.6, w_feat=0.4, tau_joint=0.85)
    assert abs(policy.w_conf - 0.6) < 1e-5
    assert abs(policy.w_feat - 0.4) < 1e-5
    assert abs(policy.tau_joint - 0.85) < 1e-5


def test_confidence_feature_gating_pure_confidence_equivalence():
    """When w_feat = 0, policy must match pure confidence thresholding."""
    policy = ConfidenceFeatureGatingPolicy(w_conf=1.0, w_feat=0.0, tau_joint=0.80)
    model = build_resnet18(pretrained=False, num_classes=1)
    model.eval()

    x = torch.randn(10, 3, 96, 96)
    decision = policy.select(model, x)

    assert isinstance(decision, ConfFeatureDecision)
    assert decision.mc_passes_executed == 1
    # Check that keep_mask matches S_conf >= 0.80
    expected_mask = decision.conf_scores >= 0.80
    assert torch.equal(decision.keep_mask, expected_mask)


def test_confidence_feature_gating_with_512d_centroids():
    """Policy must compute cosine distance against 512-D prototypes."""
    centroids = torch.randn(2, 512)
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=-1)

    policy = ConfidenceFeatureGatingPolicy(
        w_conf=0.5, w_feat=0.5, tau_joint=0.75, source_centroids=centroids
    )
    model = build_resnet18(pretrained=False, num_classes=1)
    model.eval()

    x = torch.randn(8, 3, 96, 96)
    decision = policy.select(model, x)

    assert decision.conf_scores.shape == (8,)
    assert decision.feat_scores.shape == (8,)
    assert decision.joint_scores.shape == (8,)
    assert decision.cosine_distances.shape == (8,)
    assert (decision.cosine_distances >= 0.0).all()
    assert (decision.cosine_distances <= 2.0).all()
    assert (decision.joint_scores >= 0.0).all()
    assert (decision.joint_scores <= 1.0).all()
