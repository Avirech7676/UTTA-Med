"""Source-only training loop. Early-stops on OOD validation only — never on target test."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.classification import binary_metrics, brier, ece


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device=None, max_batches: Optional[int] = None) -> Dict[str, float]:
    device = device or _device()
    model.eval()
    ys, ps = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.view(-1).float()
        logits = model(x).view(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.cpu().numpy())
        ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    metrics = binary_metrics(y_true, y_prob)
    metrics["ece"] = ece(y_true, y_prob)
    metrics["brier"] = brier(y_true, y_prob)
    metrics["n"] = int(len(y_true))
    return metrics


def train_source(
    model,
    train_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    checkpoint_path: Optional[Path] = None,
    device=None,
    use_amp: bool = True,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> Dict:
    device = device or _device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    amp_ok = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_ok)

    best_auroc = -1.0
    stale = 0
    history = []
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for i, batch in enumerate(pbar):
            if max_train_batches is not None and i >= max_train_batches:
                break
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = y.view(-1, 1).float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_ok):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device=device, max_batches=max_eval_batches)
        row = {"epoch": epoch, "train_loss": running / max(n_seen, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"epoch {epoch:02d}  loss={row['train_loss']:.4f}  "
            f"val_auroc={val_metrics['auroc']:.4f}  val_f1={val_metrics['f1']:.4f}  "
            f"val_ece={val_metrics['ece']:.4f}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": best_state,
                        "epoch": epoch,
                        "val_auroc": best_auroc,
                        "amp": amp_ok,
                    },
                    checkpoint_path,
                )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch} (best val AUROC={best_auroc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_auroc": best_auroc,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "amp": amp_ok,
        "device": str(device),
    }
