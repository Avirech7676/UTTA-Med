"""
UTTA-Med 2.1: Class-Aware Feature-Distance Gating Module
=========================================================
Implements class-conditional manifold distance and head-feature agreement verification.

Unlike class-agnostic gating (which accepts samples close to ANY class prototype),
Class-Aware Feature Gating enforces two joint requirements:
  1. Manifold Alignment: The penultimate representation z must be closer to the centroid
     of the PREDICTED class (c_{y_hat}) than to competing classes (c_{1 - y_hat}).
  2. Prototype Proximity: The distance d(z, c_{y_hat}) must lie within verified source support.

Mathematical Formulation:
  - Predictive probability: p = softmax(logits)
  - Predicted class: y_hat = argmax(p)
  - Distance to predicted class prototype: d_pred = dist(z, c_{y_hat})
  - Distance to alternate class prototype: d_alt = dist(z, c_{1 - y_hat})
  - Manifold margin: M(z) = d_alt - d_pred
  - Head-Manifold Agreement: Agree(x) = (M(z) > 0)
  - Class-Aware Feature Score:
      S_class_feat(x) = [1.0 / (1.0 + d_pred)] * sigmoid(M(z) / tau_margin)
  - Convex Joint Score:
      S_joint(x) = w_conf * S_conf(x) + w_feat * S_class_feat(x)
  - Admission Rule:
      Admit(x) <===> S_joint(x) >= tau_joint AND Agree(x) == True
"""

from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassAwareFeatureGatingPolicy:
    """
    Class-Aware Feature Distance Gating for Test-Time Adaptation.
    """
    def __init__(
        self,
        w_conf: float = 0.50,
        w_feat: float = 0.50,
        tau_joint: float = 0.80,
        tau_margin: float = 0.20,
        distance_metric: str = "cosine",
        enforce_margin_agreement: bool = True,
        num_classes: int = 2,
        embed_dim: int = 512,
    ):
        if not np.isclose(w_conf + w_feat, 1.0, atol=1e-5):
            raise ValueError(f"Weights w_conf ({w_conf}) and w_feat ({w_feat}) must sum to 1.0.")
        if distance_metric not in ("cosine", "normalized_euclidean"):
            raise ValueError(f"Unsupported distance_metric: {distance_metric}. Must be 'cosine' or 'normalized_euclidean'.")

        self.w_conf = float(w_conf)
        self.w_feat = float(w_feat)
        self.tau_joint = float(tau_joint)
        self.tau_margin = float(tau_margin)
        self.distance_metric = distance_metric
        self.enforce_margin_agreement = bool(enforce_margin_agreement)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)

    def compute_distances(
        self,
        features: Union[torch.Tensor, np.ndarray],
        centroids: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes pairwise distances between features (B, D) and class centroids (C, D).
        Returns:
          dists: (B, C) array of distances
          feat_norm: (B, D) L2-normalized feature representations
        """
        if isinstance(features, torch.Tensor):
            feat_arr = features.detach().cpu().float().numpy()
        else:
            feat_arr = np.asarray(features, dtype=np.float32)

        if isinstance(centroids, torch.Tensor):
            cent_arr = centroids.detach().cpu().float().numpy()
        else:
            cent_arr = np.asarray(centroids, dtype=np.float32)

        if feat_arr.ndim == 1:
            feat_arr = feat_arr[np.newaxis, :]
        if cent_arr.ndim == 1:
            cent_arr = cent_arr[np.newaxis, :]

        if feat_arr.shape[1] != self.embed_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.embed_dim}, got {feat_arr.shape[1]}. "
                "Feature representation must not be truncated."
            )
        if cent_arr.shape[1] != self.embed_dim:
            raise ValueError(
                f"Centroid dimension mismatch: expected {self.embed_dim}, got {cent_arr.shape[1]}."
            )
        if cent_arr.shape[0] != self.num_classes:
            raise ValueError(
                f"Centroid class count mismatch: expected {self.num_classes}, got {cent_arr.shape[0]}."
            )

        # L2-normalize
        feat_norm = feat_arr / (np.linalg.norm(feat_arr, axis=1, keepdims=True) + 1e-8)
        cent_norm = cent_arr / (np.linalg.norm(cent_arr, axis=1, keepdims=True) + 1e-8)

        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_sim
            sim = np.dot(feat_norm, cent_norm.T) # (B, C)
            sim = np.clip(sim, -1.0, 1.0)
            dists = 1.0 - sim
        else:
            # Normalized Euclidean distance
            # ||z - c||^2 / 2 = 1 - z.c
            sim = np.dot(feat_norm, cent_norm.T)
            sim = np.clip(sim, -1.0, 1.0)
            dists = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - sim)))

        return dists, feat_norm

    def evaluate(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        features: Union[torch.Tensor, np.ndarray],
        centroids: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluates sample reliability and admission decisions.
        """
        if isinstance(logits, torch.Tensor):
            logits_arr = logits.detach().cpu().float().numpy()
        else:
            logits_arr = np.asarray(logits, dtype=np.float32)

        if logits_arr.ndim == 1:
            logits_arr = logits_arr[np.newaxis, :]

        # Softmax probabilities
        exp_l = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        dists, _ = self.compute_distances(features, centroids)
        B = logits_arr.shape[0]

        pred_classes = np.argmax(probs, axis=1) # (B,)
        conf_scores = np.max(probs, axis=1)     # (B,) in [0.5, 1.0] for binary

        # Normalized confidence score in [0, 1]
        s_conf = 2.0 * (conf_scores - 0.5)

        # Distance to predicted class centroid
        d_pred = np.array([dists[i, pred_classes[i]] for i in range(B)], dtype=np.float32)

        # Distance to alternate class centroid (for binary: 1 - pred_class)
        alt_classes = 1 - pred_classes
        d_alt = np.array([dists[i, alt_classes[i]] for i in range(B)], dtype=np.float32)

        # Nearest centroid class
        nearest_centroid = np.argmin(dists, axis=1)

        # Manifold margin: M(z) = d_alt - d_pred
        margins = d_alt - d_pred
        head_manifold_agreement = (margins > 0.0)

        # Sigmoid scaling on margin
        margin_factor = 1.0 / (1.0 + np.exp(-margins / (self.tau_margin + 1e-8)))

        # Prototype proximity score
        proximity_score = 1.0 / (1.0 + d_pred)

        # Class-aware feature reliability score
        s_class_feat = proximity_score * margin_factor

        # Convex combination
        s_joint = self.w_conf * s_conf + self.w_feat * s_class_feat

        # Admission condition
        if self.enforce_margin_agreement:
            admit_mask = (s_joint >= self.tau_joint) & head_manifold_agreement
        else:
            admit_mask = (s_joint >= self.tau_joint)

        return {
            "admit_mask": admit_mask,
            "joint_scores": s_joint,
            "conf_scores": conf_scores,
            "class_feat_scores": s_class_feat,
            "margins": margins,
            "d_pred": d_pred,
            "d_alt": d_alt,
            "head_manifold_agreement": head_manifold_agreement,
            "pred_classes": pred_classes,
            "nearest_centroid_classes": nearest_centroid,
            "coverage": float(np.mean(admit_mask)),
        }
