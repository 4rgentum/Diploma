"""Calibration utilities — temperature scaling + threshold selection.

Implements Guo et al. (2017) ``T``-scaling with L-BFGS on validation
logits, plus two threshold selectors (target-FPR, F1-max).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaler optimised by L-BFGS."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def apply_numpy(self, logits: np.ndarray) -> np.ndarray:
        t = float(self.temperature.clamp_min(1e-3).item())
        return logits.astype(np.float64) / t

    def fit(self, logits: np.ndarray, labels: np.ndarray, *, max_iter: int = 100) -> "TemperatureScaler":
        """Minimise NLL on the val logits w.r.t. T."""
        x = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
        # Re-init to 1.0 each fit — keeps things idempotent across runs.
        self.temperature.data.fill_(1.0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        criterion = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            t = self.temperature.clamp_min(1e-3)
            loss = criterion(x / t, y)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except RuntimeError:
            # L-BFGS occasionally fails on tiny inputs; fallback to grid.
            self._grid_fit(x, y)
        return self

    def _grid_fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        with torch.no_grad():
            best_t, best_loss = 1.0, float("inf")
            for t in np.linspace(0.5, 5.0, 91):
                p = torch.sigmoid(x / t).clamp(1e-6, 1 - 1e-6)
                loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean().item()
                if loss < best_loss:
                    best_loss, best_t = loss, float(t)
            self.temperature.data.fill_(best_t)


def find_threshold_for_target_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_fpr: float = 0.01,
) -> float:
    """Smallest threshold such that ``FPR(threshold) <= target_fpr``.

    Computed in O(N log N) via sorted score traversal — no quadratic scan.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    neg_scores = np.sort(y_score[y_true == 0])
    if neg_scores.size == 0:
        return 0.5
    # FPR(t) = (number of negatives >= t) / N_neg.
    # We want the smallest t with FPR(t) <= target.
    n_neg = neg_scores.size
    max_false_positives = int(np.floor(target_fpr * n_neg))
    if max_false_positives >= n_neg:
        return float(neg_scores.min())
    idx = n_neg - max_false_positives - 1
    return float(neg_scores[idx]) + 1e-9  # strictly above this negative


def find_threshold_for_f1_max(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_candidates: int = 201,
) -> float:
    """Threshold that maximises F1 on the given data."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0:
        return 0.5
    grid = np.linspace(y_score.min(), y_score.max(), n_candidates)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = float(f1_score(y_true, (y_score >= t).astype(np.int64), zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
