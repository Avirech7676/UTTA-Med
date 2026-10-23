"""Modular SourceTrainer for UTTA-Med model training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.classification import binary_metrics, brier, ece

logger = logging.getLogger(__name__)


class SourceTrainer:
    """
    Standard supervised trainer for source domain models on Camelyon17.

    Features:
    - 1-logit binary BCE model (official architecture freeze)
    - Comprehensive clinical & calibration metrics (AUROC, AUPRC, F1, ECE, Brier)
    - Early stopping on OOD validation set (center 1) — never target test
    - Checkpoint saving and reload verification
    - Detailed loss tracking for learning dynamics verification
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_dir: Optional[str | Path] = None,
        patience: int = 5,
        metric_for_best: str = "auroc",
        use_amp: bool = False,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = bool(use_amp and (self.device.type == "cuda"))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.patience = patience
        self.metric_for_best = metric_for_best

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_score: float = -1.0
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.history: List[Dict[str, Any]] = []

    def train_epoch(
        self,
        epoch: int,
        max_batches: Optional[int] = None,
    ) -> Tuple[float, List[float]]:
        """
        Execute one training epoch.

        Returns:
            avg_loss: Average training loss across seen batches
            batch_losses: List of individual batch losses for learning trend verification
        """
        self.model.train()
        running_loss = 0.0
        n_samples = 0
        batch_losses: List[float] = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar):
            if max_batches is not None and i >= max_batches:
                break

            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)
            batch_size = images.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(images)
                target = labels.view(-1, 1).float()
                loss = self.criterion(logits.view(-1, 1), target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = float(loss.item())
            batch_losses.append(loss_val)
            running_loss += loss_val * batch_size
            n_samples += batch_size

            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = running_loss / max(n_samples, 1)
        return avg_loss, batch_losses

    @torch.no_grad()
    def evaluate(
        self,
        loader: Optional[DataLoader] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on specified loader (default: self.val_loader).

        Returns comprehensive classification and calibration metrics.
        """
        eval_loader = loader if loader is not None else self.val_loader
        self.model.eval()

        all_y_true: List[np.ndarray] = []
        all_y_prob: List[np.ndarray] = []
        total_eval_loss = 0.0
        n_samples = 0

        for i, batch in enumerate(eval_loader):
            if max_batches is not None and i >= max_batches:
                break

            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)
            batch_size = images.size(0)

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(images)
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                target = labels.view(-1, 1).float()
                loss = self.criterion(logits.view(-1, 1), target)

            total_eval_loss += float(loss.item()) * batch_size
            n_samples += batch_size

            all_y_true.append(labels.view(-1).cpu().numpy())
            all_y_prob.append(probs)

        if not all_y_true:
            return {"loss": float("nan"), "n": 0}

        y_true = np.concatenate(all_y_true).astype(int)
        y_prob = np.concatenate(all_y_prob).astype(float)

        metrics = binary_metrics(y_true, y_prob)
        metrics["loss"] = float(total_eval_loss / max(n_samples, 1))
        metrics["ece"] = float(ece(y_true, y_prob))
        metrics["brier"] = float(brier(y_true, y_prob))
        metrics["n"] = int(len(y_true))

        return metrics

    def train(
        self,
        epochs: int = 10,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop across epochs with validation and early stopping.
        """
        stale_epochs = 0
        all_train_batch_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            train_loss, batch_losses = self.train_epoch(
                epoch=epoch,
                max_batches=max_train_batches,
            )
            all_train_batch_losses.extend(batch_losses)

            val_metrics = self.evaluate(
                loader=self.val_loader,
                max_batches=max_val_batches,
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_record)

            score = val_metrics.get(self.metric_for_best, float("nan"))
            if np.isnan(score):
                # Fall back to accuracy or f1 if AUROC is undefined on degenerate batch
                score = val_metrics.get("f1", 0.0)

            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics.get('loss', 0.0):.4f} | "
                f"Val AUROC: {val_metrics.get('auroc', 0.0):.4f} | "
                f"Val F1: {val_metrics.get('f1', 0.0):.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f}"
            )

            is_best = self.best_state_dict is None or score > self.best_score
            if is_best:
                self.best_score = score
                self.best_epoch = epoch
                stale_epochs = 0
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                if self.checkpoint_dir:
                    self.save_checkpoint(
                        self.checkpoint_dir / "checkpoint.pt",
                        epoch=epoch,
                        metrics=val_metrics,
                    )
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best {self.metric_for_best}: {self.best_score:.4f} at epoch {self.best_epoch})"
                    )
                    break

        # Load best state dict if available
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "train_batch_losses": all_train_batch_losses,
        }

    def save_checkpoint(
        self,
        filepath: str | Path,
        epoch: int,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save training checkpoint to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_score": self.best_score,
        }
        if extra:
            payload.update(extra)

        torch.save(payload, filepath)
        return filepath

    def load_checkpoint(self, filepath: str | Path) -> Dict[str, Any]:
        """Load checkpoint and restore model and optimizer state."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
