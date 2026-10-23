"""Multi-Class Evidential ResNet-18 for Computational Pathology.

Formulates Evidential Deep Learning (EDL) via Subjective Logic and
Dirichlet Prior Networks on 9-class colorectal cancer histopathology
(NCT-CRC-HE-100K and CRC-VAL-HE-7K).

Mathematical Formulation:
- Non-negative evidence: e = softplus(f_θ(x)) >= 0
- Dirichlet parameters: α_c = e_c + 1
- Dirichlet strength: S = sum_{c=1}^K α_c
- Expected probability: p_c = α_c / S
- Belief mass: b_c = e_c / S = (α_c - 1) / S
- Epistemic uncertainty (vacuity): u = K / S
Subjective Logic constraint: sum_{c=1}^K b_c + u = 1.0 (exact).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NCT_CLASSES = [
    "ADI",   # 0: Adipose
    "BACK",  # 1: Background
    "DEB",   # 2: Debris
    "LYM",   # 3: Lymphocytes
    "MUC",   # 4: Mucus
    "MUS",   # 5: Smooth muscle
    "NORM",  # 6: Normal colon mucosa
    "STR",   # 7: Cancer-associated stroma
    "TUM",   # 8: Colorectal adenocarcinoma epithelium
]
NUM_CLASSES = len(NCT_CLASSES)  # 9


class EvidentialResNet18(nn.Module):
    """ResNet-18 with Dirichlet Evidential output head."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_p: float = 0.5,
        pretrained: bool = False,
        evidence_activation: str = "softplus",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.evidence_activation = evidence_activation

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Extract stem (conv1 to avgpool)
        self.stem = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        return torch.flatten(feat, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward returns logits for compatibility."""
        return self.forward_logits(x)

    def forward_evidence(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute full evidential quantities under Dirichlet Prior parameterization.

        Returns:
            dict containing:
            - logits: raw linear logits z
            - evidence: non-negative evidence e
            - alpha: Dirichlet concentration parameters α = e + 1
            - S: Dirichlet precision S = sum(α)
            - prob: expected categorical probability p = α / S
            - belief: subjective belief masses b = e / S
            - vacuity: epistemic uncertainty u = K / S
            - confidence: max expected probability max(p)
            - entropy: Shannon entropy of expected probability
        """
        logits = self.forward_logits(x)
        return compute_evidential_quantities(
            logits, activation=self.evidence_activation, num_classes=self.num_classes
        )


def compute_evidential_quantities(
    logits: torch.Tensor,
    activation: str = "softplus",
    num_classes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pure functional transformation from logits to Dirichlet evidential terms."""
    if num_classes is None:
        num_classes = logits.shape[-1]

    if activation == "softplus":
        evidence = F.softplus(logits)
    elif activation == "relu":
        evidence = F.relu(logits)
    elif activation == "exp":
        evidence = torch.exp(torch.clamp(logits, max=10.0))
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Dirichlet parameters α = e + 1
    alpha = evidence + 1.0

    # Total Dirichlet precision S = sum(α)
    S = torch.sum(alpha, dim=-1, keepdim=True)

    # Expected probability p = α / S
    prob = alpha / S

    # Subjective belief mass b = e / S
    belief = evidence / S

    # Epistemic uncertainty (vacuity) u = K / S
    vacuity = float(num_classes) / S.squeeze(-1)

    # Softmax / expected confidence
    confidence, _ = torch.max(prob, dim=-1)

    # Expected Shannon entropy H(p)
    eps = 1e-8
    entropy = -torch.sum(prob * torch.log(prob + eps), dim=-1)

    return {
        "logits": logits,
        "evidence": evidence,
        "alpha": alpha,
        "S": S.squeeze(-1),
        "prob": prob,
        "belief": belief,
        "vacuity": vacuity,
        "confidence": confidence,
        "entropy": entropy,
    }


def build_evidential_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    dropout_p: float = 0.5,
) -> EvidentialResNet18:
    """Builder for 9-class evidential ResNet-18."""
    return EvidentialResNet18(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
    )
