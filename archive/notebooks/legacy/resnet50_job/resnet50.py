"""ResNet-50 backbone (optional capacity check B2 / B9). Same 1-logit head as ResNet-18."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_p: float = 0.5, num_classes: int = 1):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def enable_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_predict(self, x: torch.Tensor, n_passes: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        self.enable_dropout()
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                probs.append(self.predict_proba(x).squeeze(-1))
        stacked = torch.stack(probs, dim=0)
        return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)


def build_resnet50(pretrained: bool = True, dropout_p: float = 0.5) -> ResNet50Classifier:
    return ResNet50Classifier(pretrained=pretrained, dropout_p=dropout_p)
