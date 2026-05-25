"""Distribution-shift indicators.

* PSI — Population Stability Index. Threshold 0.25 = significant drift
  (Fiddler AI, Arize AI guidelines).
* KL — symmetric variant for binned distributions.
* MMD — RBF kernel with median heuristic bandwidth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import spatial


def psi(ref: np.ndarray, cur: np.ndarray, *, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two 1-D distributions."""
    ref = np.asarray(ref).astype(np.float64).ravel()
    cur = np.asarray(cur).astype(np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    # Quantile-based bin edges from the reference; ensures the bins are well populated.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = _dedup_strict(edges)
    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def kl_divergence(ref: np.ndarray, cur: np.ndarray, *, n_bins: int = 20, eps: float = 1e-6) -> float:
    """KL(P || Q) for binned reference / current distributions."""
    ref = np.asarray(ref).astype(np.float64).ravel()
    cur = np.asarray(cur).astype(np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), n_bins + 1)
    edges = _dedup_strict(edges)
    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def mmd_rbf(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    """MMD^2 with Gaussian kernel, bandwidth via median heuristic."""
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref).astype(np.float64)
    cur = np.asarray(cur).astype(np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)
    if ref.shape[0] == 0 or cur.shape[0] == 0:
        return 0.0

    if ref.shape[0] > max_samples:
        ref = ref[rng.choice(ref.shape[0], max_samples, replace=False)]
    if cur.shape[0] > max_samples:
        cur = cur[rng.choice(cur.shape[0], max_samples, replace=False)]

    # Median heuristic on the joint pairwise distances.
    joint = np.vstack([ref, cur])
    sub = joint if joint.shape[0] <= 500 else joint[rng.choice(joint.shape[0], 500, replace=False)]
    dist = spatial.distance.pdist(sub, metric="euclidean")
    sigma = float(np.median(dist)) if dist.size > 0 else 1.0
    if sigma < 1e-6:
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)

    def _gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = spatial.distance.cdist(a, b, metric="sqeuclidean")
        return np.exp(-gamma * d)

    kxx = _gram(ref, ref).mean()
    kyy = _gram(cur, cur).mean()
    kxy = _gram(ref, cur).mean()
    return float(kxx + kyy - 2 * kxy)


def drift_report(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    psi_threshold: float = 0.25,
    column_names: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-feature PSI/KL and one MMD score on the whole array."""
    reference = np.asarray(reference)
    current = np.asarray(current)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if current.ndim == 1:
        current = current.reshape(-1, 1)

    n_features = reference.shape[1]
    per_feature_psi = np.zeros(n_features, dtype=np.float64)
    per_feature_kl = np.zeros(n_features, dtype=np.float64)
    for i in range(n_features):
        per_feature_psi[i] = psi(reference[:, i], current[:, i])
        per_feature_kl[i] = kl_divergence(reference[:, i], current[:, i])

    mmd_val = mmd_rbf(reference, current)

    n_over = int((per_feature_psi > psi_threshold).sum())
    drift_alarm = bool(per_feature_psi.mean() > psi_threshold) or bool(n_over >= max(1, n_features // 10))

    feature_psi: dict[str, float]
    if column_names and len(column_names) == n_features:
        feature_psi = {column_names[i]: float(per_feature_psi[i]) for i in range(n_features)}
    else:
        feature_psi = {f"f{i}": float(per_feature_psi[i]) for i in range(n_features)}

    return {
        "psi": {
            "mean": float(per_feature_psi.mean()),
            "max": float(per_feature_psi.max()) if n_features > 0 else 0.0,
            "n_features_over_threshold": n_over,
            "per_feature": feature_psi,
        },
        "kl": {
            "mean": float(per_feature_kl.mean()),
            "max": float(per_feature_kl.max()) if n_features > 0 else 0.0,
        },
        "mmd": float(mmd_val),
        "drift_alarm": drift_alarm,
        "threshold": float(psi_threshold),
    }


def _dedup_strict(edges: np.ndarray) -> np.ndarray:
    """Ensure strictly increasing bin edges (constant-feature defence)."""
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
        else:
            out.append(out[-1] + 1e-9)
    return np.array(out, dtype=np.float64)
