"""Experiment E3 — temperature scaling and target-FPR threshold calibration
applied to the deep models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.eval import (  # noqa: E402
    TemperatureScaler,
    binary_metrics,
    expected_calibration_error,
    find_threshold_for_target_fpr,
)
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.utils import ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.12_E3")
MODELS = ["cnn_lstm", "bilstm", "tcn", "transformer", "lstm", "gru", "cnn1d", "mlp"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "E3_calibration.csv")
    args = parser.parse_args()
    ensure_dir(args.out.parent)

    val_npz = np.load(args.data / "val_windows.npz", allow_pickle=True)
    test_npz = np.load(args.data / "test_windows.npz", allow_pickle=True)
    X_val, y_val = val_npz["windows"], val_npz["labels"]
    X_test, y_test = test_npz["windows"], test_npz["labels"]

    rows = []
    for name in MODELS:
        ckpt = ROOT / "models" / f"{name}_seed{args.seed}.pt"
        cfg_path = ROOT / "configs" / "models" / f"{name}.yaml"
        if not ckpt.exists() or not cfg_path.exists():
            LOG.warning("Missing artefacts for %s — skip", name)
            continue
        cfg = load_yaml(cfg_path)
        cfg.setdefault("input_dim", int(X_val.shape[-1]))
        cfg.setdefault("window", int(X_val.shape[1]))
        model = build_model(cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu").get("state_dict"))
        model.eval()
        p_val = model.predict_proba(X_val).astype(np.float64)
        p_test = model.predict_proba(X_test).astype(np.float64)

        eps = 1e-6
        logits_val = np.log(np.clip(p_val, eps, 1 - eps) / np.clip(1 - p_val, eps, 1 - eps))
        ece_before = expected_calibration_error(y_val, p_val)
        scaler = TemperatureScaler()
        scaler.fit(logits_val.astype(np.float32), y_val.astype(np.float32))
        T = float(scaler.temperature.detach().item())
        p_val_cal = 1.0 / (1.0 + np.exp(-logits_val / max(T, 1e-3)))
        ece_after = expected_calibration_error(y_val, p_val_cal)

        logits_test = np.log(np.clip(p_test, eps, 1 - eps) / np.clip(1 - p_test, eps, 1 - eps))
        p_test_cal = 1.0 / (1.0 + np.exp(-logits_test / max(T, 1e-3)))

        tau = find_threshold_for_target_fpr(y_val, p_val_cal, target_fpr=args.target_fpr)
        m = binary_metrics(y_test, p_test_cal, threshold=tau)
        rows.append(
            {
                "model": name,
                "T": T,
                "tau": tau,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "ece_reduction_pct": float(100 * (ece_before - ece_after) / max(ece_before, 1e-9)),
                "test_fpr": m["fpr"],
                "test_f1": m["f1"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
            }
        )

    df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    df.to_csv(args.out, index=False)
    LOG.info("E3 -> %s\n%s", args.out, df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
