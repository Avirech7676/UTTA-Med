"""Unit tests for Optimal Transport Prototype Alignment (OT-TTA)."""
import torch
import torch.nn as nn
from src.tta.ot_tta import PrototypeOT_TTA, sinkhorn_knopp_transport


class MockModelWithFeatures(nn.Module):
    def __init__(self, in_features=64, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, in_features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, num_classes)

    def extract_features(self, x):
        h = self.pool(self.bn(self.conv(x)))
        return torch.flatten(h, 1)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.fc(feats)


def test_sinkhorn_knopp():
    cost = torch.tensor([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
    ])
    priors = torch.tensor([0.5, 0.5])
    Q = sinkhorn_knopp_transport(cost, priors, epsilon=0.05, max_iters=20)
    assert Q.shape == (4, 2)
    # Row sum must equal 1/B = 0.25
    row_sums = Q.sum(dim=1)
    assert torch.allclose(row_sums, torch.full((4,), 0.25), atol=1e-3)
    # Column sum must equal priors = 0.5
    col_sums = Q.sum(dim=0)
    assert torch.allclose(col_sums, priors, atol=1e-2)


def test_prototype_ot_adaptation():
    model = MockModelWithFeatures(in_features=32, num_classes=2)
    prototypes = torch.randn(2, 32)
    controller = PrototypeOT_TTA(
        model,
        source_prototypes=prototypes,
        lr=1e-4,
        max_displacement=0.05,
    )

    x = torch.randn(4, 3, 32, 32)
    res = controller.adapt_batch(x)
    assert res["adapted"]
    assert "loss_total" in res
    assert "loss_ot" in res
    assert "loss_div" in res
    assert res["logits"].shape == (4, 2)
