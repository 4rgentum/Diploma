"""Metric computations for binary NIDS classification.

Everything works on plain numpy arrays — no torch dependency, so the same code
runs in train, eval, agent runs and unit tests. Computations follow
scikit-learn definitions; we keep our own thin wrappers to control behaviour
on degenerate inputs (single-class predictions, empty arrays, ties).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    """Return a dict of standard binary metrics.

    Defensive against single-class inputs — degenerate metrics yield 0.0
    rather than crashing.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0:
        return {k: 0.0 for k in ("precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc")}

    y_pred = (y_score >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, pos_label=1
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0

    # AUCs require both classes present.
    classes_present = np.unique(y_true)
    if classes_present.size > 1:
        roc = float(roc_auc_score(y_true, y_score))
        pr = float(average_precision_score(y_true, y_score))
    else:
        roc, pr = 0.0, 0.0

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "mcc": float(mcc),
        "roc_auc": roc,
        "pr_auc": pr,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Standard ECE with equal-width bins (Guo et al., 2017)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    if y_true.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = float(y_true.size)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = float((y_true[mask] == (y_prob[mask] >= 0.5).astype(np.int64)).mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def bootstrap_ci(
    metric_fn: Any,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_resamples: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(point, low, high)`` for the bootstrap distribution of ``metric_fn``."""
    rng = np.random.default_rng(seed)
    n = y_true.size
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(metric_fn(y_true, y_score))
    samples: list[float] = []
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, n)
        try:
            samples.append(float(metric_fn(y_true[idx], y_score[idx])))
        except (ValueError, ZeroDivisionError):
            continue
    if not samples:
        return point, point, point
    low_q = (1 - ci) / 2
    high_q = 1 - low_q
    low, high = np.quantile(samples, [low_q, high_q])
    return point, float(low), float(high)


def fp_per_time(
    timestamps: np.ndarray,
    is_fp: np.ndarray,
    *,
    bin_seconds: float = 60.0,
) -> float:
    """Average number of false positives per ``bin_seconds`` of observation."""
    timestamps = np.asarray(timestamps)
    is_fp = np.asarray(is_fp).astype(bool)
    if timestamps.size == 0:
        return 0.0
    duration = float(timestamps.max() - timestamps.min())
    if duration <= 0:
        return float(is_fp.sum())
    bins = max(1.0, duration / bin_seconds)
    return float(is_fp.sum()) / bins


def time_to_detect(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Median time-to-detect across attack episodes.

    An "episode" is a maximal run of consecutive samples with ``y_true == 1``.
    TTD for an episode is the time between the first attack timestamp and the
    first ``y_pred == 1`` inside the episode. Episodes never detected
    contribute the episode duration plus one bin to keep the metric finite.
    """
    timestamps = np.asarray(timestamps).astype(np.float64)
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if y_true.size == 0:
        return {"median_ttd": 0.0, "n_episodes": 0, "detected_episodes": 0}

    starts = np.flatnonzero(np.diff(np.concatenate([[0], y_true])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([y_true, [0]])) == -1)
    ttds: list[float] = []
    detected = 0
    for s, e in zip(starts, ends, strict=False):
        idx = np.flatnonzero(y_pred[s : e + 1])
        if idx.size > 0:
            detected += 1
            ttds.append(float(timestamps[s + idx[0]] - timestamps[s]))
        else:
            ttds.append(float(timestamps[e] - timestamps[s]) + 1.0)
    if not ttds:
        return {"median_ttd": 0.0, "n_episodes": 0, "detected_episodes": 0}
    return {
        "median_ttd": float(np.median(ttds)),
        "n_episodes": int(len(ttds)),
        "detected_episodes": int(detected),
    }
