"""Attention Multiple Instance Learning (Attn-MIL) for Slide-Level Histopathology.

Implements Gated Attention-MIL (Ilse et al., ICML 2018 / CLAM, Lu et al., Nature BME 2021)
for Whole-Slide Image (WSI) metastasis detection and test-time attention adaptation.

Mathematical Formulation:
1. Gated Attention Pooling:
   a_i = softmax(w^T (tanh(V h_i) ⊙ sigm(U h_i)))
2. Slide Representation:
   h_slide = sum_{i=1}^M a_i * h_i
3. Slide Staging Logit:
   y_slide = W_s * h_slide + b_s
4. Test-Time Attention Sharpening:
   Adapts attention projection matrices (V, U, w) at test time to focus on true
   metastatic foci while preventing attention dispersion over benign lymphoid tissue.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Gated Attention-MIL model for aggregating patch embeddings into WSI-level predictions."""

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_p: float = 0.25,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Gated Attention Network (Ilse et al., 2018)
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # Slide-level classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def compute_attention(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized attention coefficients over M patches.
        Args:
            h: Patch embeddings tensor of shape (M, in_features) or (B, M, in_features).
        Returns:
            Attention weights tensor of shape (M, 1) or (B, M, 1).
        """
        if h.dim() == 2:
            # Single slide: (M, in_features)
            v = self.attention_V(h)  # (M, hidden_dim)
            u = self.attention_U(h)  # (M, hidden_dim)
            raw = self.attention_weights(v * u)  # (M, 1)
            attn = F.softmax(raw, dim=0)  # (M, 1)
            return attn
        elif h.dim() == 3:
            # Batch of slides: (B, M, in_features)
            v = self.attention_V(h)  # (B, M, hidden_dim)
            u = self.attention_U(h)  # (B, M, hidden_dim)
            raw = self.attention_weights(v * u)  # (B, M, 1)
            attn = F.softmax(raw, dim=1)  # (B, M, 1)
            return attn
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {h.shape}")

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing slide-level logits and patch attention maps.
        Returns:
            (slide_logits, attention_weights)
        """
        if h.dim() == 2:
            attn = self.compute_attention(h)  # (M, 1)
            slide_feat = torch.sum(attn * h, dim=0, keepdim=True)  # (1, in_features)
            logits = self.classifier(slide_feat)  # (1, num_classes)
            return logits, attn
        else:
            attn = self.compute_attention(h)  # (B, M, 1)
            slide_feat = torch.sum(attn * h, dim=1)  # (B, in_features)
            logits = self.classifier(slide_feat)  # (B, num_classes)
            return logits, attn


class TestTimeAttentionAdapter(nn.Module):
    """Test-time adaptation controller for Attention-MIL whole-slide staging."""
    __test__ = False

    def __init__(
        self,
        mil_model: GatedAttentionMIL,
        lr: float = 1e-4,
        sparsity_weight: float = 0.1,
        anchor_weight: float = 0.01,
    ):
        super().__init__()
        self.mil_model = mil_model
        self.lr = lr
        self.sparsity_weight = sparsity_weight
        self.anchor_weight = anchor_weight

        # Freeze the slide-level classifier; adapt ONLY the attention mechanism (V, U, w)
        for p in self.mil_model.classifier.parameters():
            p.requires_grad_(False)

        for p in self.mil_model.attention_V.parameters():
            p.requires_grad_(True)
        for p in self.mil_model.attention_U.parameters():
            p.requires_grad_(True)
        for p in self.mil_model.attention_weights.parameters():
            p.requires_grad_(True)

        self.adaptable_params = [
            p for p in self.mil_model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)

        # Snapshot immutable anchor
        self.anchor = {
            id(p): p.data.clone().detach() for p in self.adaptable_params
        }

    def adapt_slide(self, h_patches: torch.Tensor) -> Dict[str, Any]:
        """
        Adapts the attention pooling mechanism on a single Whole Slide Image.
        Args:
            h_patches: (M, in_features) embeddings of all tissue tiles from the slide.
        """
        self.mil_model.eval()

        # Compute initial attention
        logits, attn = self.mil_model(h_patches)  # attn: (M, 1)
        probs = F.softmax(logits, dim=-1)

        # 1. Prediction entropy loss
        loss_entropy = -torch.sum(probs * torch.log(probs + 1e-8))

        # 2. Attention Sparsity Loss (encourage peak focus on tumor foci, penalize uniform blur)
        # Herfindahl-Hirschman index penalty: -sum(a_i^2) pushes attention to be peaked
        loss_sparsity = -torch.sum(attn ** 2)

        # 3. Source Anchor Loss
        loss_anchor = torch.tensor(0.0, device=h_patches.device)
        for p in self.adaptable_params:
            if id(p) in self.anchor:
                loss_anchor = loss_anchor + torch.sum((p - self.anchor[id(p)].to(h_patches.device)) ** 2)

        total_loss = loss_entropy + self.sparsity_weight * loss_sparsity + self.anchor_weight * loss_anchor

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Post-adaptation slide prediction
        with torch.no_grad():
            post_logits, post_attn = self.mil_model(h_patches)
            post_probs = F.softmax(post_logits, dim=-1)

        return {
            "slide_logits": post_logits.detach(),
            "slide_probs": post_probs.detach(),
            "attention": post_attn.detach(),
            "loss": float(total_loss.item()),
        }
