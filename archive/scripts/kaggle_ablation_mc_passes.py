"""Standalone Kaggle Runner for Ablation Study A: MC-Dropout Pass Sensitivity.

Run on Kaggle with GPU T4, Internet ON.
Input: Dataset containing camelyon17_resnet18_source_s42_BEST.pt
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_dataset
from tqdm.auto import tqdm

WORK = Path("/kaggle/working" if os.path.exists("/kaggle") else "./results/ablations/mc_passes")
WORK.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

PASS_GRID = [5, 10, 20, 30, 50]
REF_PASS = 20
TTA_LR = 1e-5
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=None)
        d = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.classifier = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(d, 1)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def find_checkpoint() -> str:
    hits = glob.glob("/kaggle/input/**/*s42*BEST*.pt", recursive=True)
    hits += glob.glob("/kaggle/input/**/*source_s42*.pt", recursive=True)
    hits += glob.glob("checkpoints/*s42*BEST*.pt")
    hits += glob.glob("**/camelyon17_resnet18_source_s42_BEST.pt", recursive=True)
    if not hits:
        raise FileNotFoundError("Could not find camelyon17_resnet18_source_s42_BEST.pt")
    print("Found checkpoint:", hits[0])
    return hits[0]


def bn_affine_params(m: nn.Module) -> List[nn.Parameter]:
    for p in m.parameters():
        p.requires_grad = False
    ps = []
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d) and mod.affine:
            mod.weight.requires_grad = True
            mod.bias.requires_grad = True
            ps.extend([mod.weight, mod.bias])
    return ps


def eval_mode(m: nn.Module):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def mc_mode(m: nn.Module):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
        elif isinstance(mod, nn.Dropout):
            mod.train()


def tta_mode(m: nn.Module):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.eval()
        elif isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def entropy(z: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(z).clamp(1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).view(-1)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p <= bins[i + 1]) if i == n_bins - 1 else (p >= bins[i]) & (p < bins[i + 1])
        if mask.any():
            e += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(e)


def metrics_of(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob).ravel()
    pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    n_neg = max(int((y == 0).sum()), 1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / n_neg),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "ece": ece_score(y, p),
        "brier": float(np.mean((p - y) ** 2)),
    }


def flip_table(y_true: np.ndarray, p_before: np.ndarray, p_after: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true).ravel().astype(int)
    b = (np.asarray(p_before).ravel() >= 0.5).astype(int)
    a = (np.asarray(p_after).ravel() >= 0.5).astype(int)

    correct_before = b == y
    correct_after = a == y
    wrong_before = ~correct_before
    harmful = int((correct_before & ~correct_after).sum())
    correction = int((wrong_before & correct_after).sum())
    n_correct_before = max(int(correct_before.sum()), 1)
    n_wrong_before = max(int(wrong_before.sum()), 1)

    return {
        "correction_rate": float(correction / n_wrong_before),
        "harm_rate": float(harmful / n_correct_before),
        "flip_rate": float((b != a).mean()),
        "correction": correction,
        "harmful": harmful,
        "stable_correct": int((correct_before & correct_after).sum()),
        "persistent_wrong": int((wrong_before & ~correct_after).sum()),
    }


@torch.no_grad()
def mc_predict(m: Net, x: torch.Tensor, n_mc: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mc_mode(m)
    ps = [torch.sigmoid(m(x)).view(-1) for _ in range(n_mc)]
    P = torch.stack(ps, 0)
    return P.mean(0), P.var(0, unbiased=False)


def run_pipeline():
    ckpt_path = find_checkpoint()
    print("Checkpoint verified:", ckpt_path)


if __name__ == "__main__":
    run_pipeline()
