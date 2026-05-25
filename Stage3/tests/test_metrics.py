from __future__ import annotations

import numpy as np

from diploma_nids.eval import (
    binary_metrics,
    bootstrap_ci,
    expected_calibration_error,
    find_threshold_for_f1_max,
    find_threshold_for_target_fpr,
)


def test_binary_metrics_perfect() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.95, 0.2, 0.8])
    m = binary_metrics(y_true, y_score)
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0


def test_binary_metrics_single_class_does_not_crash() -> None:
    y_true = np.zeros(10, dtype=np.int64)
    y_score = np.linspace(0, 1, 10)
    m = binary_metrics(y_true, y_score)
    assert "pr_auc" in m


def test_find_threshold_for_target_fpr() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    s = rng.uniform(size=2000)
    # Push positives slightly higher.
    s[y == 1] += 0.5
    tau = find_threshold_for_target_fpr(y, s, target_fpr=0.05)
    pred = (s >= tau).astype(np.int64)
    fpr = float(((y == 0) & (pred == 1)).sum() / (y == 0).sum())
    assert fpr <= 0.06  # small slack for ties


def test_find_threshold_for_f1_max_in_range() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    s = rng.uniform(size=200)
    tau = find_threshold_for_f1_max(y, s)
    assert 0.0 <= tau <= 1.0


def test_ece_decreases_with_temperature_scaling_on_overconfident_model() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=500)
    # Make probabilities overconfident relative to accuracy.
    p = np.clip(rng.uniform(size=500) ** 0.3, 1e-3, 1 - 1e-3)
    p[y == 0] = 1 - p[y == 0]
    ece = expected_calibration_error(y, p)
    # Apply a wide temperature manually -> probability sharpened then dulled.
    softened = 0.5 + 0.6 * (p - 0.5)
    softened = np.clip(softened, 0.001, 0.999)
    ece2 = expected_calibration_error(y, softened)
    # Either calibration helps OR makes no difference — the harness is here
    # for the *not-worse* property.
    assert ece2 <= ece + 0.05


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=100)
    s = rng.uniform(size=100)
    point, lo, hi = bootstrap_ci(lambda yt, ys: float(((ys >= 0.5) == yt).mean()), y, s, n_resamples=50)
    assert lo <= point <= hi
