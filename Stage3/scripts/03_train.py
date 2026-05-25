"""Train any registered model from a YAML config.

Examples:
    python scripts/03_train.py --model configs/models/cnn_lstm.yaml \\
        --train configs/train/full.yaml --seed 42

    python scripts/03_train.py --model configs/models/random_forest.yaml \\
        --classical --seed 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.training import TrainerConfig, train_classical, train_deep  # noqa: E402
from diploma_nids.utils import dump_json, ensure_dir, get_logger, load_yaml, set_seed  # noqa: E402

LOG = get_logger("scripts.03_train")
CLASSICAL_NAMES = {"logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="model YAML")
    parser.add_argument("--train", type=Path, default=ROOT / "configs" / "train" / "default.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "experiments" / "runs")
    args = parser.parse_args()

    set_seed(args.seed)
    train_cfg = load_yaml(args.train)
    model_cfg = load_yaml(args.model)

    # Load processed windows.
    train_npz = np.load(args.data / "train_windows.npz", allow_pickle=True)
    val_npz = np.load(args.data / "val_windows.npz", allow_pickle=True)
    X_tr, y_tr = train_npz["windows"], train_npz["labels"]
    X_val, y_val = val_npz["windows"], val_npz["labels"]

    name = model_cfg["name"]
    is_classical = name in CLASSICAL_NAMES
    model_cfg.setdefault("input_dim", int(X_tr.shape[-1]))
    model_cfg.setdefault("window", int(X_tr.shape[1]))
    if is_classical:
        # Override the seed in the params for deterministic baselines.
        model_cfg.setdefault("params", {})
        model_cfg["params"].setdefault("random_state", args.seed)
    model = build_model(model_cfg)

    ensure_dir(args.out)
    suffix = "joblib" if is_classical else "pt"
    ckpt_path = args.out / f"{name}_seed{args.seed}.{suffix}"

    if is_classical:
        LOG.info("Training classical %s (seed=%d)", name, args.seed)
        meta = train_classical(model, X_tr, y_tr, save_path=ckpt_path)
        history = {"fit_time_sec": meta["fit_time_sec"], "type": "classical"}
    else:
        LOG.info("Training deep %s (seed=%d)", name, args.seed)
        cfg = TrainerConfig(**train_cfg)
        history = train_deep(model, X_tr, y_tr, X_val, y_val, cfg=cfg, save_path=ckpt_path)
        history = asdict(history)
        history["type"] = "deep"

    ensure_dir(args.runs_dir)
    run_path = args.runs_dir / f"{name}_seed{args.seed}_train.json"
    dump_json({"model": name, "seed": args.seed, "history": history, "checkpoint": str(ckpt_path)}, run_path)
    LOG.info("Saved checkpoint -> %s", ckpt_path)
    LOG.info("Saved run log -> %s", run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
