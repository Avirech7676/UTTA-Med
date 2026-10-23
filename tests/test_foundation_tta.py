"""Unit tests for Foundation Model LoRA Test-Time Adaptation."""
import torch
import torch.nn as nn
from src.models.foundation_tta import FoundationModelLoRAAdapter, LoRALinear


class MockViTBackbone(nn.Module):
    """Simple ViT mock with attention projections."""
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_stem = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_stem(x)
        h = h.flatten(2).transpose(1, 2)  # (B, N, D)
        q = self.q_proj(h)
        v = self.v_proj(h)
        out = self.fc1(q + v)
        return out.mean(dim=1)  # (B, D)


def test_lora_linear_forward():
    orig_linear = nn.Linear(32, 64)
    lora = LoRALinear(orig_linear, rank=4, alpha=8.0)
    x = torch.randn(2, 32)
    out = lora(x)
    assert out.shape == (2, 64)
    # Original weights frozen
    assert not lora.weight.requires_grad
    # Adapter weights trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_foundation_adapter_freezing():
    backbone = MockViTBackbone(embed_dim=64)
    adapter = FoundationModelLoRAAdapter(backbone, num_classes=2, lora_rank=4)

    # Backbone weights must all be frozen
    for p in adapter.backbone.conv_stem.parameters():
        assert not p.requires_grad

    # Only LoRA weights and head are trainable
    trainable_params = adapter.get_trainable_parameters()
    assert len(trainable_params) > 0
    for p in trainable_params:
        assert p.requires_grad


def test_foundation_adapter_empty_batch_skipping():
    backbone = MockViTBackbone(embed_dim=64)
    adapter = FoundationModelLoRAAdapter(backbone, num_classes=2, lora_rank=4)
    x = torch.randn(4, 3, 32, 32)

    # Threshold set to 1.0 (no sample eligible)
    res = adapter.adapt_batch(x, confidence_threshold=1.0)
    assert not res["adapted"]
    assert res["eligible"] == 0


def test_foundation_adapter_bounded_projection():
    backbone = MockViTBackbone(embed_dim=64)
    adapter = FoundationModelLoRAAdapter(backbone, num_classes=2, lora_rank=4, max_displacement=0.02)
    x = torch.randn(4, 3, 32, 32)

    # Record initial weights
    initial_A = adapter.lora_layers["q_proj"].lora_A.clone()

    # Adapt on low-threshold batch
    res = adapter.adapt_batch(x, confidence_threshold=0.0)
    assert res["adapted"]

    # Verify displacement bounded by max_displacement
    post_A = adapter.lora_layers["q_proj"].lora_A
    diff = (post_A - initial_A).abs()
    assert diff.max().item() <= 0.02 + 1e-6
