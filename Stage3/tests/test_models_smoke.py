from __future__ import annotations

import numpy as np
import pytest
import torch

from diploma_nids.models import available_models, build_model
from diploma_nids.models.classical import _ClassicalAdapter

DL_MODELS = ["mlp", "cnn1d", "lstm", "gru", "bilstm", "tcn", "transformer", "cnn_lstm", "autoencoder", "vae"]
CLASSICAL_MODELS = ["logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm"]


def test_registry_contains_all_models() -> None:
    names = set(available_models().keys())
    expected = set(DL_MODELS + CLASSICAL_MODELS)
    assert expected.issubset(names), f"missing: {expected - names}"


@pytest.mark.parametrize("name", DL_MODELS)
def test_deep_forward_smoke(name: str) -> None:
    cfg = {"name": name, "input_dim": 16, "window": 16, "params": {}}
    model = build_model(cfg)
    model.eval()
    x = torch.randn(4, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,) or out.shape == (4, 1)


@pytest.mark.parametrize("name", CLASSICAL_MODELS)
def test_classical_fit_predict(name: str) -> None:
    cfg = {"name": name, "params": {"n_estimators": 10} if name == "xgboost" else {}}
    model = build_model(cfg)
    assert isinstance(model, _ClassicalAdapter)
    X = np.random.RandomState(0).normal(size=(64, 8, 4)).astype(np.float32)
    y = np.random.RandomState(0).randint(0, 2, size=64).astype(np.int64)
    model.fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (64,)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_cnn_lstm_marked_proposed() -> None:
    meta = available_models()
    assert meta["cnn_lstm"].get("proposed") is True
