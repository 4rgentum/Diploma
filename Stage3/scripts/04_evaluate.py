"""Evaluate a trained model on val and test, with temperature scaling and
target-FPR threshold calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.eval import (  # noqa: E402
    TemperatureScaler,
    binary_metrics,
    expected_calibration_error,
    find_threshold_for_target_fpr,
    per_class_recall,
)
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.models.classical import _ClassicalAdapter  # noqa: E402, WPS437
from diploma_nids.utils import dump_json, ensure_dir, get_logger, load_yaml, set_seed  # noqa: E402

LOG = get_logger("scripts.04_evaluate")
CLASSICAL_NAMES = {"logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm"}


def _load_classical(path: Path):
    return _ClassicalAdapter.load(path)


def _load_deep(model_cfg: dict, ckpt: Path, input_dim: int, window: int):
    model_cfg.setdefault("input_dim", input_dim)
    model_cfg.setdefault("window", window)
    model = build_model(model_cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--eval-cfg", type=Path, default=ROOT / "configs" / "eval" / "default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "runs")
    args = parser.parse_args()

    set_seed(args.seed)
    model_cfg = load_yaml(args.model)
    eval_cfg = load_yaml(args.eval_cfg)

    val_npz = np.load(args.data / "val_windows.npz", allow_pickle=True)
    test_npz = np.load(args.data / "test_windows.npz", allow_pickle=True)
    X_val, y_val = val_npz["windows"], val_npz["labels"]
    X_test, y_test = test_npz["windows"], test_npz["labels"]
    test_cat = test_npz["attack_cat"]

    name = model_cfg["name"]
    is_classical = name in CLASSICAL_NAMES
    if is_classical:
        model = _load_classical(args.checkpoint)
    else:
        model = _load_deep(model_cfg, args.checkpoint, int(X_val.shape[-1]), int(X_val.shape[1]))

    p_val = model.predict_proba(X_val).astype(np.float64)
    p_test = model.predict_proba(X_test).astype(np.float64)

    # Temperature scaling (deep only — classical probs are already calibrated
    # by their estimators).
    T = 1.0
    ece_before = expected_calibration_error(y_val, p_val, n_bins=int(eval_cfg.get("ece_n_bins", 15)))
    if not is_classical:
        eps = 1e-6
        logits_val = np.log(np.clip(p_val, eps, 1 - eps) / np.clip(1 - p_val, eps, 1 - eps))
        scaler = TemperatureScaler()
        scaler.fit(logits_val.astype(np.float32), y_val.astype(np.float32))
        T = float(scaler.temperature.detach().cpu().item())
        p_val = 1.0 / (1.0 + np.exp(-logits_val / max(T, 1e-3)))
        logits_test = np.log(np.clip(p_test, eps, 1 - eps) / np.clip(1 - p_test, eps, 1 - eps))
        p_test = 1.0 / (1.0 + np.exp(-logits_test / max(T, 1e-3)))
    ece_after = expected_calibration_error(y_val, p_val)

    tau = find_threshold_for_target_fpr(y_val, p_val, target_fpr=float(eval_cfg.get("target_fpr", 0.01)))
    LOG.info("Calibrated threshold τ = %.4f (T=%.3f, ECE %.4f -> %.4f)", tau, T, ece_before, ece_after)

    test_metrics = binary_metrics(y_test, p_test, threshold=tau)
    val_metrics = binary_metrics(y_val, p_val, threshold=tau)

    pc_recall = per_class_recall(y_test, (p_test >= tau).astype(np.int64), test_cat)

    ensure_dir(args.out)
    out_payload = {
        "model": name,
        "checkpoint": str(args.checkpoint),
        "threshold": tau,
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "val": val_metrics,
        "test": test_metrics,
        "per_class_recall": pc_recall.to_dict(orient="records"),
    }
    out_path = args.out / f"{name}_seed{args.seed}_eval.json"
    dump_json(out_payload, out_path)
    LOG.info("Eval -> %s", out_path)
    LOG.info(
        "TEST: F1=%.4f PR-AUC=%.4f FPR=%.4f Recall=%.4f Precision=%.4f",
        test_metrics["f1"],
        test_metrics["pr_auc"],
        test_metrics["fpr"],
        test_metrics["recall"],
        test_metrics["precision"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
