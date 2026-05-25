"""Generic trainer for the deep models + helpers for the classical models.

Design:

* ``Trainer`` is a small object holding the model, optimiser, loss and
  a config dataclass. ``fit(train_loader, val_loader)`` runs the loop,
  records per-epoch metrics, and returns a ``TrainHistory`` dict the
  scripts persist to ``experiments/runs/<model>_<seed>_train.json``.

* Reproducibility is non-negotiable — every numeric source is seeded by
  the caller via ``utils.seed.set_seed`` *before* the trainer is built.

* Early stopping watches ``val_pr_auc`` by default; PR-AUC is the
  Stage 1 §6 nominal optimisation target on the imbalanced binary task.

* ``train_classical`` is the parallel helper for classical models —
  it fits a ``_ClassicalAdapter`` and returns the same metric dict.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..eval.metrics import binary_metrics
from ..models.autoencoder import Autoencoder, VAE
from ..models.base import BaseDeepModel, BaseModel
from .losses import build_loss


@dataclass
class TrainerConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_min_lr: float = 1e-6
    early_stopping_patience: int = 7
    early_stopping_metric: str = "val_pr_auc"
    early_stopping_mode: str = "max"
    loss: dict[str, Any] = field(default_factory=lambda: {"name": "focal", "alpha": 0.25, "gamma": 2.0})
    num_workers: int = 0
    log_every: int = 50
    use_amp: bool = False  # CPU build — keep disabled.


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_value: float = -math.inf
    total_time_sec: float = 0.0
    early_stopped: bool = False


# --------------------------------------------------------------------------- #
# Deep trainer                                                                 #
# --------------------------------------------------------------------------- #


class Trainer:
    def __init__(
        self,
        model: BaseDeepModel,
        config: TrainerConfig,
        *,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.loss_fn = build_loss(self.cfg.loss)
        self.optim = self._make_optimizer()
        self.sched = self._make_scheduler(self.optim)
        self.is_unsupervised_ae = isinstance(model, (Autoencoder, VAE)) and not model.is_supervised

    # ---- public API -------------------------------------------------------- #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    ) -> TrainHistory:
        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)

        # For unsupervised AE / VAE, training data should only include normals.
        if self.is_unsupervised_ae:
            mask = (y_train == 0)
            if mask.sum() < 16:
                raise RuntimeError("Not enough normal samples for unsupervised AE")
            train_loader = self._loader(X_train[mask], y_train[mask], shuffle=True)

        history = TrainHistory()
        wall_start = time.perf_counter()
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_probs, val_labels = self._eval_epoch(val_loader)
            metrics = binary_metrics(val_labels, val_probs)
            metrics.update({"train_loss": float(train_loss), "val_loss": float(val_loss)})
            lr = self.optim.param_groups[0]["lr"]

            history.train_loss.append(float(train_loss))
            history.val_loss.append(float(val_loss))
            history.val_pr_auc.append(float(metrics.get("pr_auc", 0.0)))
            history.val_roc_auc.append(float(metrics.get("roc_auc", 0.0)))
            history.val_f1.append(float(metrics.get("f1", 0.0)))
            history.lr.append(float(lr))

            current = float(metrics.get(self.cfg.early_stopping_metric, 0.0))
            improved = (
                current > history.best_value
                if self.cfg.early_stopping_mode == "max"
                else current < history.best_value
            )
            if improved:
                history.best_value = current
                history.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            # Calibration buffer for AE/VAE — set after every epoch so the
            # checkpoint at any time reflects the latest scoring scale.
            if self.is_unsupervised_ae:
                self._refresh_ae_calibration(X_val[y_val == 0])

            if on_epoch is not None:
                on_epoch(epoch, metrics)

            if self.sched is not None:
                self.sched.step()

            if bad_epochs >= self.cfg.early_stopping_patience:
                history.early_stopped = True
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        history.total_time_sec = time.perf_counter() - wall_start
        return history

    # ---- internals --------------------------------------------------------- #

    def _make_optimizer(self) -> optim.Optimizer:
        params = self.model.parameters()
        if self.cfg.optimizer.lower() == "adamw":
            return optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if self.cfg.optimizer.lower() == "adam":
            return optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        raise ValueError(f"unknown optimizer {self.cfg.optimizer!r}")

    def _make_scheduler(self, opt: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
        if self.cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg.epochs, eta_min=self.cfg.scheduler_min_lr
            )
        if self.cfg.scheduler in ("none", None):
            return None
        raise ValueError(f"unknown scheduler {self.cfg.scheduler!r}")

    def _loader(self, X: np.ndarray, y: np.ndarray, *, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            self.optim.zero_grad(set_to_none=True)

            if self.is_unsupervised_ae:
                recon, target = self.model.reconstruct(xb)  # type: ignore[union-attr]
                loss = ((recon - target) ** 2).mean()
                if isinstance(self.model, VAE):
                    mu = self.model._last_mu
                    logvar = self.model._last_logvar
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    loss = loss + self.model.kl_weight * kld
            else:
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)

            loss.backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
            self.optim.step()

            bs = xb.size(0)
            total += float(loss.item()) * bs
            count += bs
        return total / max(count, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total = 0.0
        count = 0
        all_probs: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            if self.is_unsupervised_ae:
                # AE forward already returns logit form; cross-entropy reuses target=0 for normals
                target = (yb > 0).float()
            else:
                target = yb.float()
            loss = self.loss_fn(logits, target) if not self.is_unsupervised_ae else (
                ((self.model.reconstruct(xb)[0] - xb.flatten(1)) ** 2).mean()
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_y.append(yb.cpu().numpy())
            bs = xb.size(0)
            total += float(loss.item()) * bs
            count += bs
        return total / max(count, 1), np.concatenate(all_probs), np.concatenate(all_y)

    @torch.no_grad()
    def _refresh_ae_calibration(self, X_normal: np.ndarray) -> None:
        if not isinstance(self.model, Autoencoder):
            return
        if X_normal.size == 0:
            return
        self.model.eval()
        x = torch.from_numpy(np.ascontiguousarray(X_normal, dtype=np.float32))
        recon, target = self.model.reconstruct(x)
        err = ((recon - target) ** 2).mean(dim=1)
        ref = float(err.quantile(0.99).item())
        self.model.ref_score_max = torch.tensor(max(ref, 1e-6))


# --------------------------------------------------------------------------- #
# Classical helpers                                                            #
# --------------------------------------------------------------------------- #


def train_deep(
    model: BaseDeepModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: TrainerConfig,
    save_path: str | Path | None = None,
) -> TrainHistory:
    tr = Trainer(model, cfg)
    history = tr.fit(X_train, y_train, X_val, y_val)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, save_path)
    return history


def train_classical(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit a classical adapter and persist via its ``save`` method."""
    start = time.perf_counter()
    model.fit(X_train, y_train)  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - start
    if save_path is not None:
        model.save(save_path)  # type: ignore[attr-defined]
    return {"fit_time_sec": elapsed}
