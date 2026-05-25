"""Experiment E6 — operational performance.

Measures per-window inference latency (CPU, batch=1) and per-batch
throughput (CPU, batch=256). Median of 50 timed iterations after a 5-call
warmup, as required by Stage 2 §8.7.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.models.classical import _ClassicalAdapter  # noqa: E402, WPS437
from diploma_nids.utils import ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.15_E6")

DEEP_MODELS = ("mlp", "cnn1d", "lstm", "gru", "bilstm", "tcn", "transformer", "cnn_lstm", "autoencoder", "vae")
CLASSICAL_MODELS = ("logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm")


def _bench(model, X1: np.ndarray, Xb: np.ndarray, *, warmup: int = 5, iters: int = 50) -> dict[str, float]:
    for _ in range(warmup):
        model.predict_proba(X1)
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        model.predict_proba(X1)
        samples.append(time.perf_counter() - t)
    lat_ms = 1000.0 * float(np.median(samples))

    for _ in range(warmup):
        model.predict_proba(Xb)
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        model.predict_proba(Xb)
        samples.append(time.perf_counter() - t)
    throughput = Xb.shape[0] / float(np.median(samples))
    return {"latency_ms": lat_ms, "throughput_eps": throughput}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "E6_latency.csv")
    args = parser.parse_args()
    ensure_dir(args.out.parent)

    test_npz = np.load(args.data / "test_windows.npz", allow_pickle=True)
    X = test_npz["windows"][: args.batch_size + 8]
    X1 = X[:1]
    Xb = X[: args.batch_size]

    rows = []
    for name in DEEP_MODELS + CLASSICAL_MODELS:
        ckpt_path = ROOT / "models" / f"{name}_seed{args.seed}.{'joblib' if name in CLASSICAL_MODELS else 'pt'}"
        if not ckpt_path.exists():
            LOG.warning("Skip %s — no checkpoint at %s", name, ckpt_path)
            continue
        if name in CLASSICAL_MODELS:
            model = _ClassicalAdapter.load(ckpt_path)
        else:
            cfg = load_yaml(ROOT / "configs" / "models" / f"{name}.yaml")
            cfg.setdefault("input_dim", int(X.shape[-1]))
            cfg.setdefault("window", int(X.shape[1]))
            model = build_model(cfg)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu").get("state_dict"))
            model.eval()
        try:
            bench = _bench(model, X1, Xb)
            rows.append({"model": name, **bench, "nfr3_met": bench["latency_ms"] <= 50.0, "nfr4_met": bench["throughput_eps"] >= 1000.0})
            LOG.info("%-20s lat=%.3f ms  throughput=%.1f EPS", name, bench["latency_ms"], bench["throughput_eps"])
        except Exception as exc:
            LOG.error("Bench failed for %s: %s", name, exc)

    df = pd.DataFrame(rows).sort_values("latency_ms")
    df.to_csv(args.out, index=False)
    LOG.info("E6 -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
