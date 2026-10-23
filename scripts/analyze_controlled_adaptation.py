#!/usr/bin/env python3
"""Stage 6: Controlled Adaptation Hyperparameter Optimization.

Sweeps:
- Adaptation learning rate: [1e-3, 1e-4, 1e-5, 1e-6]
- Consistency regularization lambda: [0.0, 0.1, 0.5]
- Elastic source anchoring lambda: [0.0, 0.01, 0.1]
- Gradient clipping: [None, 1.0]

Optimized strictly on Camelyon17-WILDS Center 1 validation partition.
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import json
from copy import deepcopy
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.resnet18 import build_resnet18

def load_center1_val_patches(seed: int = 42):
    parquet_path = Path("c:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data/validation-00002-of-00003.parquet")
    pf = pq.ParquetFile(parquet_path)
    tbl_0 = pf.read_row_group(0, columns=["image", "label"])
    tbl_1 = pf.read_row_group(115, columns=["image", "label"])
    
    combined = []
    for tbl in [tbl_0, tbl_1]:
        df = tbl.to_pandas()
        for _, row in df.iterrows():
            lbl = int(row["label"])
            b = row["image"]["bytes"]
            im = Image.open(io.BytesIO(b)).convert("RGB")
            arr = (np.array(im, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            combined.append((torch.from_numpy(arr).permute(2, 0, 1).float(), lbl))
            
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    imgs = [item[0] for item in combined]
    labels = [item[1] for item in combined]
    return torch.stack(imgs, dim=0), np.array(labels)

def evaluate_adaptation_run(
    base_model: nn.Module,
    x: torch.Tensor,
    y: np.ndarray,
    lr: float = 1e-4,
    lambda_cons: float = 0.0,
    lambda_anchor: float = 0.0,
    clip_norm: float = 1.0,
    gating_thresh: float = 0.90,
    batch_size: int = 16,
) -> dict:
    device = torch.device("cpu")
    model = deepcopy(base_model).to(device)
    
    # Configure trainable parameters: only BatchNorm affine parameters (official TTA protocol)
    model.eval()
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            params.append(m.weight)
            params.append(m.bias)
        else:
            for p in m.parameters():
                p.requires_grad = False

    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Cache initial source weights for elastic anchoring
    source_anchors = {id(p): p.data.clone() for p in params}

    initial_preds = []
    adapted_preds = []
    updates_taken = 0
    rollbacks_taken = 0
    
    n_batches = int(np.ceil(len(x) / batch_size))

    for b_idx in range(n_batches):
        xb = x[b_idx * batch_size : (b_idx + 1) * batch_size].to(device)
        
        # 1. Deterministic inference before update
        with torch.no_grad():
            model.eval()
            out_before = model(xb)
            p_before = torch.sigmoid(out_before).view(-1)
            initial_preds.append(p_before.numpy())

        # 2. Confidence Gating Selection
        conf = torch.maximum(p_before, 1.0 - p_before)
        selected_mask = conf > gating_thresh
        
        if selected_mask.sum() == 0:
            adapted_preds.append(p_before.numpy())
            continue

        xb_sel = xb[selected_mask]
        
        # Cache state before adaptation for rollback safety
        state_backup = {k: v.clone() for k, v in model.state_dict().items()}
        opt_backup = deepcopy(optimizer.state_dict())

        # Forward pass for adaptation
        out = model(xb_sel)
        p = torch.sigmoid(out).view(-1)
        
        # Unsupervised entropy loss: - [p log p + (1-p) log (1-p)]
        eps = 1e-7
        p_clamped = torch.clamp(p, eps, 1.0 - eps)
        loss_ent = - (p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped)).mean()
        
        loss = loss_ent
        
        # Consistency regularization under rotation
        if lambda_cons > 0.0:
            x_rot = torch.rot90(xb_sel, 1, dims=[2, 3])
            out_rot = model(x_rot)
            p_rot = torch.sigmoid(out_rot).view(-1)
            loss_cons = F.mse_loss(p, p_rot)
            loss = loss + lambda_cons * loss_cons

        optimizer.zero_grad()
        loss.backward()

        # Elastic parameter anchoring gradient
        if lambda_anchor > 0.0:
            for p_param in params:
                if p_param.grad is not None:
                    diff = p_param.data - source_anchors[id(p_param)]
                    p_param.grad.data.add_(lambda_anchor * diff)

        # Gradient clipping
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)

        optimizer.step()
        
        # Predict on entire batch with adapted model
        with torch.no_grad():
            out_after = model(xb)
            p_after = torch.sigmoid(out_after).view(-1)
            
            # Safety check: if entropy explodes or NaN occurs, execute rollback
            if torch.isnan(p_after).any() or (p_after.min() == p_after.max()):
                model.load_state_dict(state_backup)
                optimizer.load_state_dict(opt_backup)
                rollbacks_taken += 1
                adapted_preds.append(p_before.numpy())
            else:
                updates_taken += 1
                adapted_preds.append(p_after.numpy())

    p_init = np.concatenate(initial_preds)
    p_adp = np.concatenate(adapted_preds)

    f1_init = f1_score(y, (p_init >= 0.5).astype(int), zero_division=0)
    f1_adp = f1_score(y, (p_adp >= 0.5).astype(int), zero_division=0)
    auc_init = roc_auc_score(y, p_init)
    auc_adp = roc_auc_score(y, p_adp)
    acc_init = accuracy_score(y, (p_init >= 0.5).astype(int))
    acc_adp = accuracy_score(y, (p_adp >= 0.5).astype(int))

    # Harm rate: samples correctly classified before that are flipped to incorrect
    y_init_bin = (p_init >= 0.5).astype(int)
    y_adp_bin = (p_adp >= 0.5).astype(int)
    harm = float(np.mean((y == y_init_bin) & (y != y_adp_bin)))

    return {
        "lr": lr,
        "lambda_cons": lambda_cons,
        "lambda_anchor": lambda_anchor,
        "clip_norm": clip_norm,
        "updates": updates_taken,
        "rollbacks": rollbacks_taken,
        "initial_f1": round(float(f1_init), 4),
        "adapted_f1": round(float(f1_adp), 4),
        "delta_f1": round(float(f1_adp - f1_init), 4),
        "initial_auroc": round(float(auc_init), 4),
        "adapted_auroc": round(float(auc_adp), 4),
        "delta_auroc": round(float(auc_adp - auc_init), 4),
        "harm_rate": round(float(harm), 4),
    }

def main():
    print("[*] Loading Center 1 validation cohort...")
    x, y = load_center1_val_patches(seed=42)
    print(f"[*] Loaded {len(x)} validation patches.")

    ckpt_path = Path("checkpoints/camelyon17_resnet18_source_s42_BEST.pt")
    base_model = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    base_model.load_state_dict(state_dict)
    base_model.eval()

    print("\n" + "=" * 90)
    print("STAGE 6: CONTROLLED ADAPTATION HYPERPARAMETER OPTIMIZATION (Camelyon17 Center 1)")
    print("=" * 90)
    print(f"{'LR':<8} | {'l_cons':<8} | {'l_anch':<8} | {'Clip':<6} | {'Upd':<4} | {'Roll':<4} | {'Init F1':<8} | {'Adp F1':<8} | {'d_F1':<8} | {'Harm':<6}")
    print("-" * 90)

    configs = [
        # 1. LR sweeps (baseline TENT style)
        {"lr": 1e-3, "lambda_cons": 0.0, "lambda_anchor": 0.0, "clip_norm": 1.0},
        {"lr": 1e-4, "lambda_cons": 0.0, "lambda_anchor": 0.0, "clip_norm": 1.0},
        {"lr": 5e-5, "lambda_cons": 0.0, "lambda_anchor": 0.0, "clip_norm": 1.0},
        {"lr": 1e-5, "lambda_cons": 0.0, "lambda_anchor": 0.0, "clip_norm": 1.0},
        
        # 2. Consistency Regularization sweeps (at lr=1e-4)
        {"lr": 1e-4, "lambda_cons": 0.1, "lambda_anchor": 0.0, "clip_norm": 1.0},
        {"lr": 1e-4, "lambda_cons": 0.5, "lambda_anchor": 0.0, "clip_norm": 1.0},
        
        # 3. Source Elastic Anchoring sweeps (at lr=1e-4)
        {"lr": 1e-4, "lambda_cons": 0.0, "lambda_anchor": 0.01, "clip_norm": 1.0},
        {"lr": 1e-4, "lambda_cons": 0.0, "lambda_anchor": 0.1, "clip_norm": 1.0},
        
        # 4. Synergistic Combination (UTTA-Med 2.0 configuration)
        {"lr": 1e-4, "lambda_cons": 0.1, "lambda_anchor": 0.01, "clip_norm": 1.0},
        {"lr": 5e-5, "lambda_cons": 0.1, "lambda_anchor": 0.01, "clip_norm": 1.0},
        
        # 5. Ablation: Without Gradient Clipping
        {"lr": 1e-4, "lambda_cons": 0.0, "lambda_anchor": 0.0, "clip_norm": 0.0},
    ]

    results = []
    for cfg in configs:
        res = evaluate_adaptation_run(
            base_model=base_model,
            x=x,
            y=y,
            lr=cfg["lr"],
            lambda_cons=cfg["lambda_cons"],
            lambda_anchor=cfg["lambda_anchor"],
            clip_norm=cfg["clip_norm"],
        )
        results.append(res)
        print(f"{res['lr']:<8.0e} | {res['lambda_cons']:<8.1f} | {res['lambda_anchor']:<8.2f} | {res['clip_norm']:<6.1f} | {res['updates']:<4d} | {res['rollbacks']:<4d} | {res['initial_f1']:<8.4f} | {res['adapted_f1']:<8.4f} | {res['delta_f1']:<+8.4f} | {res['harm_rate']:<6.4f}")

    out_file = Path("results/controlled_adaptation_hyperparameters.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_file}")

if __name__ == "__main__":
    main()
