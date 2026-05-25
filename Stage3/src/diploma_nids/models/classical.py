"""Classical baselines — Logistic Regression, Random Forest, XGBoost,
Isolation Forest, One-Class SVM.

All classical models share the same ``BaseModel`` API. They consume the
flattened window ``(B, W * F)`` — there is no temporal modelling here. This is
intentional: classical models in NIDS are typically applied to either single
flows or pre-aggregated feature vectors; we keep the comparison fair by
flattening the same window the deep models see.

Serialisation uses ``joblib`` so we never depend on pickle stability across
sklearn versions for our own artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

try:
    from xgboost import XGBClassifier

    _XGB_OK = True
except ModuleNotFoundError:  # pragma: no cover
    _XGB_OK = False
    XGBClassifier = None  # type: ignore[assignment]

from .base import BaseModel, register


class _ClassicalAdapter(BaseModel):
    """Wraps a scikit-style estimator so it speaks our BaseModel API."""

    name = "classical"
    accepts_windows = True
    needs_torch = False

    def __init__(self, estimator: Any, *, mode: str = "supervised", scale: bool = True) -> None:
        self.estimator = estimator
        self.mode = mode  # "supervised" | "anomaly" | "one_class"
        self.scaler: StandardScaler | None = StandardScaler() if scale else None
        self._is_fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return X.astype(np.float32, copy=False)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_ClassicalAdapter":
        X = self._flatten(X)
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        if self.mode == "supervised":
            assert y is not None, "supervised classical model requires labels"
            self.estimator.fit(X, y)
        elif self.mode in ("anomaly", "one_class"):
            # Fit only on normal samples (label == 0) if labels were provided,
            # otherwise the estimator handles unsupervised fitting itself.
            if y is not None:
                X = X[y == 0]
            self.estimator.fit(X)
        else:
            raise ValueError(f"unknown classical mode {self.mode!r}")
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self.estimator).__name__} is not fitted yet")
        X = self._flatten(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.mode == "supervised":
            if hasattr(self.estimator, "predict_proba"):
                p = self.estimator.predict_proba(X)
                # binary -> column 1; multiclass -> max across positive classes
                return (1.0 - p[:, 0] if p.shape[1] >= 2 else p[:, 0]).astype(np.float32)
            # No proba? fall back to decision_function -> sigmoid.
            score = self.estimator.decision_function(X)
            return _sigmoid(score)

        # anomaly / one_class: higher score = more anomalous.
        if hasattr(self.estimator, "score_samples"):
            # IsolationForest: higher score = less anomalous, so invert.
            s = -self.estimator.score_samples(X)
        else:
            s = -self.estimator.decision_function(X)
        # Min-max into [0, 1] using a running estimate from fit-time scores
        # if available, otherwise per-batch.
        s_min, s_max = float(np.min(s)), float(np.max(s))
        if s_max - s_min < 1e-9:
            return np.full_like(s, 0.5, dtype=np.float32)
        return ((s - s_min) / (s_max - s_min)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "scaler": self.scaler, "mode": self.mode}, p)

    @classmethod
    def load(cls, path: str | Path) -> "_ClassicalAdapter":
        payload = joblib.load(path)
        m = cls(payload["estimator"], mode=payload["mode"], scale=payload["scaler"] is not None)
        m.scaler = payload["scaler"]
        m._is_fitted = True
        return m


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #


@register("logistic_regression", family="classical")
def _build_lr(cfg: dict[str, Any]) -> _ClassicalAdapter:
    p = cfg.get("params", {})
    est = LogisticRegression(
        C=float(p.get("C", 1.0)),
        max_iter=int(p.get("max_iter", 1000)),
        solver=p.get("solver", "lbfgs"),
        n_jobs=int(p.get("n_jobs", -1)),
        class_weight=p.get("class_weight", "balanced"),
    )
    return _ClassicalAdapter(est, mode="supervised", scale=True)


@register("random_forest", family="classical")
def _build_rf(cfg: dict[str, Any]) -> _ClassicalAdapter:
    p = cfg.get("params", {})
    est = RandomForestClassifier(
        n_estimators=int(p.get("n_estimators", 200)),
        max_depth=p.get("max_depth"),
        min_samples_split=int(p.get("min_samples_split", 2)),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(p.get("random_state", 42)),
        class_weight=p.get("class_weight", "balanced"),
    )
    return _ClassicalAdapter(est, mode="supervised", scale=False)


@register("xgboost", family="classical")
def _build_xgb(cfg: dict[str, Any]) -> _ClassicalAdapter:
    if not _XGB_OK:
        raise ImportError("xgboost is not installed")
    p = cfg.get("params", {})
    est = XGBClassifier(
        n_estimators=int(p.get("n_estimators", 300)),
        max_depth=int(p.get("max_depth", 6)),
        learning_rate=float(p.get("learning_rate", 0.1)),
        subsample=float(p.get("subsample", 0.9)),
        colsample_bytree=float(p.get("colsample_bytree", 0.9)),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(p.get("random_state", 42)),
        tree_method=p.get("tree_method", "hist"),
        eval_metric="logloss",
        verbosity=0,
    )
    return _ClassicalAdapter(est, mode="supervised", scale=False)


@register("isolation_forest", family="classical", is_supervised=False)
def _build_iforest(cfg: dict[str, Any]) -> _ClassicalAdapter:
    p = cfg.get("params", {})
    est = IsolationForest(
        n_estimators=int(p.get("n_estimators", 200)),
        contamination=float(p.get("contamination", 0.1)),
        max_samples=p.get("max_samples", "auto"),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(p.get("random_state", 42)),
    )
    return _ClassicalAdapter(est, mode="anomaly", scale=True)


@register("ocsvm", family="classical", is_supervised=False)
def _build_ocsvm(cfg: dict[str, Any]) -> _ClassicalAdapter:
    p = cfg.get("params", {})
    est = OneClassSVM(
        kernel=p.get("kernel", "rbf"),
        gamma=p.get("gamma", "scale"),
        nu=float(p.get("nu", 0.1)),
    )
    return _ClassicalAdapter(est, mode="one_class", scale=True)
