"""FastAPI inference service.

Loaded lazily from environment variables so the import of the module never
fails when FastAPI is not installed (CI / minimal environments).

Endpoints:

* ``GET  /health`` — liveness.
* ``GET  /info``   — model + preprocessor + threshold metadata.
* ``POST /score``  — score a batch of raw flow records.
* ``POST /score-window`` — score an already-built window ``(W, F)``.
* ``GET  /alerts/recent?n=N`` — last N alerts emitted by AlertFormer.

Environment variables:

    DIPLOMA_MODEL_YAML       — path to model YAML config
    DIPLOMA_CHECKPOINT       — path to model checkpoint
    DIPLOMA_PREPROCESSOR     — path to JSON preprocessor
    DIPLOMA_THRESHOLD        — decision threshold (float)
    DIPLOMA_TEMPERATURE      — optional T-scaler value
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# FastAPI is optional — import lazily so the module imports cleanly even
# without it (the service script will fail fast when launched).
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel as PydanticModel
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    PydanticModel = object  # type: ignore[assignment,misc]


from ..data.preprocess import Preprocessor
from ..inference.alerting import AlertFormer
from ..inference.stream import WindowScorer
from ..models.base import build_model
from ..utils.io import load_yaml
from ..utils.logging import get_logger


def _load_model_and_state() -> tuple[Any, Preprocessor, float, str | None]:
    """Read environment, build model, return ``(model, preprocessor, tau, version)``."""
    model_yaml = os.environ.get("DIPLOMA_MODEL_YAML")
    checkpoint = os.environ.get("DIPLOMA_CHECKPOINT")
    preprocessor_path = os.environ.get("DIPLOMA_PREPROCESSOR")
    threshold = float(os.environ.get("DIPLOMA_THRESHOLD", "0.5"))
    version = os.environ.get("DIPLOMA_MODEL_VERSION")

    if not (model_yaml and checkpoint and preprocessor_path):
        raise RuntimeError(
            "Environment variables DIPLOMA_MODEL_YAML, DIPLOMA_CHECKPOINT and "
            "DIPLOMA_PREPROCESSOR must be set before launching the service."
        )

    preprocessor = Preprocessor.load(preprocessor_path)
    cfg = load_yaml(model_yaml)
    cfg.setdefault("input_dim", preprocessor.output_dim)
    cfg.setdefault("window", cfg.get("window", 32))
    model = build_model(cfg)

    if cfg["name"] in {"logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm"}:
        # Classical model — load via its own loader.
        from ..models.classical import _ClassicalAdapter  # noqa: WPS437

        model = _ClassicalAdapter.load(checkpoint)
    else:
        import torch  # local import to keep this module importable without torch.

        state = torch.load(checkpoint, map_location="cpu")
        sd = state.get("state_dict", state)
        model.load_state_dict(sd)
        model.eval()
    return model, preprocessor, threshold, version


def create_app() -> "FastAPI":  # pragma: no cover — exercised by the script.
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed; `pip install fastapi uvicorn`.")
    logger = get_logger("diploma_nids.service")

    model, preprocessor, threshold, version = _load_model_and_state()
    window = int(os.environ.get("DIPLOMA_WINDOW", "32"))
    stride = int(os.environ.get("DIPLOMA_STRIDE", "8"))
    scorer = WindowScorer(model=model, preprocessor=preprocessor, window=window, stride=stride)
    alert_former = AlertFormer(threshold=threshold, model_version=version)

    app = FastAPI(title="diploma_nids", version="0.2.0")

    class FlowBatch(PydanticModel):
        rows: list[dict[str, Any]]

    class WindowPayload(PydanticModel):
        window: list[list[float]]  # (W, F)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "threshold": threshold,
            "window": window,
            "stride": stride,
            "model_version": version,
            "feature_columns": preprocessor.state.output_columns,
        }

    @app.post("/score")
    def score(batch: FlowBatch) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for row in batch.rows:
            score_val = scorer.push(row)
            if score_val is None:
                continue
            alert = alert_former.maybe_emit(
                ts=float(row.get("timestamp", 0.0)),
                score=score_val,
                src_ip=row.get("src_ip"),
                dst_ip=row.get("dst_ip"),
                attack_cat=row.get("attack_cat"),
            )
            results.append({"score": score_val, "alert": alert.to_dict() if alert else None})
        return {"results": results, "n_alerts_total": len(alert_former.history)}

    @app.post("/score-window")
    def score_window(payload: WindowPayload) -> dict[str, Any]:
        arr = np.asarray(payload.window, dtype=np.float32)[np.newaxis, ...]
        if arr.ndim != 3:
            raise HTTPException(status_code=400, detail="window must be 2D (W, F)")
        p = float(model.predict_proba(arr)[0])
        return {"score": p, "above_threshold": bool(p >= threshold)}

    @app.get("/alerts/recent")
    def alerts_recent(n: int = 50) -> dict[str, Any]:
        items = [a.to_dict() for a in alert_former.recent(n)]
        return {"alerts": items, "count": len(items)}

    logger.info("FastAPI service ready (model=%s, threshold=%.4f)", model.__class__.__name__, threshold)
    return app
