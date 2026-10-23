"""Evidential Test-Time Adaptation (EDL-TTA) for Multi-Class Histopathology.

Implements Dirichlet Prior Network Test-Time Adaptation on 9-class
colorectal cancer tissue typing (NCT-CRC-HE).

Key Innovations:
1. Decouples Epistemic Vacuity u = K/S from Output Confidence max(p)
   on K=9 classes, breaking the binary Taylor rank-equivalence trap.
2. Hard Vacuity Gate: w(x) = 1[u(x) < τ_vac] filters out ambiguous
   or out-of-distribution tissue artifacts while adapting on certain patterns.
3. Dirichlet Bayes Risk Loss: Minimizes subjective entropy/risk on accepted samples.
4. Preserves fail-closed safety: Frozen BN running stats (bn_freeze_stats),
   empty-batch skipping, and parameter updates restricted strictly to BN affine (γ, β).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.evidential_resnet import compute_evidential_quantities
from src.tta.bn import collect_bn_affine, configure_tta_mode


class EvidentialTTA(nn.Module):
    """Evidential Test-Time Adaptation Controller."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        tau_vacuity: Optional[float] = None,
        tau_confidence: Optional[float] = None,
        tau_entropy: Optional[float] = None,
        gate_type: str = "vacuity",
        loss_type: str = "dirichlet_entropy",
        bn_mode: str = "bn_freeze_stats",
        num_classes: int = 9,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.tau_vacuity = tau_vacuity
        self.tau_confidence = tau_confidence
        self.tau_entropy = tau_entropy
        self.gate_type = gate_type
        self.loss_type = loss_type
        self.bn_mode = bn_mode
        self.num_classes = num_classes

        # Freeze all backbone and linear head weights
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Collect and enable strictly BatchNorm affine parameters (γ, β)
        self.adaptable_params = collect_bn_affine(self.model)
        for p in self.adaptable_params:
            p.requires_grad_(True)

        configure_tta_mode(self.model, bn_mode=self.bn_mode)

        # Standard TTA optimizer is Adam
        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)

        # Diagnostic counters
        self.n_batches_seen = 0
        self.n_batches_adapted = 0
        self.n_batches_skipped = 0
        self.n_samples_seen = 0
        self.n_samples_adapted = 0

    def compute_loss(
        self, alpha: torch.Tensor, S: torch.Tensor, prob: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dirichlet objective for test-time adaptation.

        Options:
        - 'dirichlet_entropy': Expected entropy under Dirichlet distribution
          E[H(p)] = ψ(S + 1) - sum_c (α_c / S) * ψ(α_c + 1)
        - 'dirichlet_bayes_risk': Subjective cross-entropy / variance risk
          sum_c p_c * (ψ(S) - ψ(α_c))
        """
        eps = 1e-8
        if self.loss_type == "dirichlet_entropy":
            # Digamma ψ function
            psi_S = torch.digamma(S + 1.0)
            psi_alpha = torch.digamma(alpha + 1.0)
            per_sample_loss = psi_S - torch.sum((alpha / S.unsqueeze(-1)) * psi_alpha, dim=-1)
        elif self.loss_type == "dirichlet_bayes_risk":
            psi_S = torch.digamma(S.unsqueeze(-1))
            psi_alpha = torch.digamma(alpha)
            per_sample_loss = torch.sum(prob * (psi_S - psi_alpha), dim=-1)
        else:
            # Fallback to standard Shannon entropy on expected probabilities
            per_sample_loss = -torch.sum(prob * torch.log(prob + eps), dim=-1)

        return per_sample_loss

    def compute_gate_mask(
        self, ev_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """Compute sample gating mask w in {0, 1}^B based on gate policy."""
        vacuity = ev_dict["vacuity"]          # shape (B,)
        confidence = ev_dict["confidence"]    # shape (B,)
        entropy = ev_dict["entropy"]          # shape (B,)

        B = vacuity.shape[0]

        if self.gate_type == "vacuity":
            if self.tau_vacuity is None:
                mask = torch.ones(B, dtype=torch.bool, device=vacuity.device)
            else:
                mask = vacuity < self.tau_vacuity

        elif self.gate_type == "confidence":
            if self.tau_confidence is None:
                mask = torch.ones(B, dtype=torch.bool, device=confidence.device)
            else:
                mask = confidence >= self.tau_confidence

        elif self.gate_type == "entropy":
            if self.tau_entropy is None:
                mask = torch.ones(B, dtype=torch.bool, device=entropy.device)
            else:
                mask = entropy <= self.tau_entropy

        elif self.gate_type == "hybrid":
            # Dual-threshold: accept if high confidence and moderate vacuity
            conf_thresh = self.tau_confidence if self.tau_confidence is not None else 0.70
            vac_thresh = self.tau_vacuity if self.tau_vacuity is not None else 0.50
            mask = (confidence >= conf_thresh) & (vacuity < vac_thresh)

        elif self.gate_type == "none":
            # Ungated (Tent equivalent)
            mask = torch.ones(B, dtype=torch.bool, device=vacuity.device)

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        coverage = float(mask.sum().item()) / float(B)
        return mask, coverage

    def adapt_batch(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Union[float, int]]]:
        """Adapt model on an unlabeled test batch x."""
        self.model.eval()
        configure_tta_mode(self.model, bn_mode=self.bn_mode)

        self.n_batches_seen += 1
        B = x.shape[0]
        self.n_samples_seen += B

        # 1. Forward pass to obtain evidential quantities
        logits = self.model(x)
        ev_dict = compute_evidential_quantities(logits, num_classes=self.num_classes)

        # 2. Compute sample gate mask
        mask, coverage = self.compute_gate_mask(ev_dict)
        n_accepted = int(mask.sum().item())

        # 3. Fail-closed: skip batch if empty
        if n_accepted == 0:
            self.n_batches_skipped += 1
            info = {
                "adapted": False,
                "n_accepted": 0,
                "coverage": 0.0,
                "loss": 0.0,
                "mean_vacuity": float(ev_dict["vacuity"].mean().item()),
                "mean_confidence": float(ev_dict["confidence"].mean().item()),
            }
            return ev_dict, info

        # 4. Compute evidential loss over accepted samples only
        per_sample_loss = self.compute_loss(
            ev_dict["alpha"], ev_dict["S"], ev_dict["prob"]
        )
        masked_loss = per_sample_loss[mask].mean()

        # 5. Gradient update
        self.optimizer.zero_grad()
        masked_loss.backward()
        self.optimizer.step()

        self.n_batches_adapted += 1
        self.n_samples_adapted += n_accepted

        info = {
            "adapted": True,
            "n_accepted": n_accepted,
            "coverage": coverage,
            "loss": float(masked_loss.item()),
            "mean_vacuity": float(ev_dict["vacuity"].mean().item()),
            "mean_confidence": float(ev_dict["confidence"].mean().item()),
        }
        return ev_dict, info


def compute_gate_rank_correlations(
    confidences: np.ndarray,
    vacuities: np.ndarray,
    entropies: np.ndarray,
) -> Dict[str, float]:
    """Calculate Spearman rank correlations among uncertainty metrics.

    Demonstrates decoupling: On multi-class problems (K=9), vacuity
    decouples from confidence (|ρ| << 0.99), proving non-redundancy.
    """
    conf = np.asarray(confidences).flatten()
    vac = np.asarray(vacuities).flatten()
    ent = np.asarray(entropies).flatten()

    # Spearman rho
    rho_conf_vac, p_cv = scipy.stats.spearmanr(conf, vac)
    rho_conf_ent, p_ce = scipy.stats.spearmanr(conf, ent)
    rho_vac_ent, p_ve = scipy.stats.spearmanr(vac, ent)

    # Invert vacuity to correlate in same direction as confidence (low vacuity = high confidence)
    rho_1mvac_conf, _ = scipy.stats.spearmanr(1.0 - vac, conf)

    return {
        "rho_conf_vs_vacuity": float(rho_conf_vac),
        "rho_conf_vs_entropy": float(rho_conf_ent),
        "rho_vacuity_vs_entropy": float(rho_vac_ent),
        "rho_1minus_vacuity_vs_conf": float(rho_1mvac_conf),
    }
