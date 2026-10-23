"""Modern Vision Backbones for Computational Pathology.

Supports ConvNeXt, Swin Transformer, and ResNet-50 with modular classification heads:
- Single-logit binary head (sigmoid)
- Two-logit head (softmax)
- Dirichlet Evidential head (Subjective Logic vacuity)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ModernHistologyClassifier(nn.Module):
    """Modular histology classifier supporting modern vision backbones."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 1,
        dropout_p: float = 0.5,
        pretrained: bool = True,
        head_type: str = "standard",  # "standard" or "evidential"
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.head_type = head_type
        self.dropout_p = dropout_p

        # Backbone instantiation
        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.embed_dim = base.fc.in_features  # 2048
        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.convnext_tiny(weights=weights)
            self.feature_extractor = base.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.embed_dim = base.classifier[2].in_features  # 768
        elif self.backbone_name == "swin_t":
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.swin_t(weights=weights)
            self.feature_extractor = base.features
            self.norm = base.norm
            self.permute = base.permute
            self.avgpool = base.avgpool
            self.embed_dim = base.head.in_features  # 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.dropout = nn.Dropout(p=dropout_p)

        # Classification Head
        out_dim = num_classes if num_classes > 1 else 1
        self.fc = nn.Linear(self.embed_dim, out_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense feature representation from input patches."""
        if self.backbone_name == "resnet50":
            feats = self.feature_extractor(x)
            feats = torch.flatten(feats, 1)
        elif self.backbone_name == "convnext_tiny":
            feats = self.feature_extractor(x)
            feats = self.avgpool(feats)
            feats = torch.flatten(feats, 1)
        elif self.backbone_name == "swin_t":
            feats = self.feature_extractor(x)
            feats = self.norm(feats)
            feats = self.permute(feats)
            feats = self.avgpool(feats)
            feats = torch.flatten(feats, 1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits or evidence."""
        feats = self.extract_features(x)
        dropped = self.dropout(feats)
        logits = self.fc(dropped)

        if self.head_type == "evidential":
            # Return non-negative Dirichlet evidence: e = softplus(z)
            return F.softplus(logits)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized prediction probabilities."""
        out = self.forward(x)
        if self.head_type == "evidential":
            alpha = out + 1.0
            s = torch.sum(alpha, dim=-1, keepdim=True)
            return alpha / s
        if self.num_classes == 1:
            return torch.sigmoid(out).squeeze(-1)
        return F.softmax(out, dim=-1)
