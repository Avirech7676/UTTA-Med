"""ResNet-50 optional backbone (B2 / B9). Same 1-logit, dropout 0.5, BCE head as ResNet-18.

Not part of the 5-seed headline table until a trained BEST.pt exists.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models

OFFICIAL_DROPOUT_P = 0.5
OFFICIAL_NUM_CLASSES = 1


class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = OFFICIAL_DROPOUT_P,
        num_classes: int = OFFICIAL_NUM_CLASSES,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                f"Official UTTA-Med head is 1-logit BCE, got num_classes={num_classes}."
            )
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
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        return self.classifier(feats), feats

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def enable_dropout(self) -> None:
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def enable_mc_dropout(self) -> None:
        self.enable_dropout()

    def disable_mc_dropout(self) -> None:
        self.eval()

    def mc_predict(self, x: torch.Tensor, n_passes: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        self.enable_dropout()
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                probs.append(self.predict_proba(x).squeeze(-1))
        stacked = torch.stack(probs, dim=0)
        return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)

    def get_bn_affine_parameters(self) -> List[nn.Parameter]:
        params = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                params.append(module.weight)
                params.append(module.bias)
        return params

    def get_bn_affine_parameter_names(self) -> List[str]:
        names = []
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                names.extend([f"{name}.weight", f"{name}.bias"])
        return names

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        bn_affine = self.get_bn_affine_parameters()
        bn_ids = {id(p) for p in bn_affine}
        return {
            "bn_affine": bn_affine,
            "other": [p for p in self.parameters() if id(p) not in bn_ids],
        }


def build_resnet50(
    pretrained: bool = True,
    dropout_p: float = OFFICIAL_DROPOUT_P,
    num_classes: int = OFFICIAL_NUM_CLASSES,
) -> ResNet50Classifier:
    return ResNet50Classifier(
        pretrained=pretrained, dropout_p=dropout_p, num_classes=num_classes
    )


ResNet50UTTAMed = ResNet50Classifier
