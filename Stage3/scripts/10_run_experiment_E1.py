"""Experiment E1 — same-conditions comparison of 14 models.

For each model, train across several seeds, evaluate on test with target-FPR
calibrated threshold, summarise in a single CSV.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.utils import ensure_dir, get_logger  # noqa: E402

LOG = get_logger("scripts.10_E1")

DEEP_MODELS = ("mlp", "cnn1d", "lstm", "gru", "bilstm", "tcn", "transformer", "cnn_lstm", "autoencoder", "vae")
CLASSICAL_MODELS = ("logistic_regression", "random_forest", "xgboost", "isolation_forest", "ocsvm")


def _run(args: list[str]) -> None:
    LOG.info("$ %s", " ".join(args))
    subprocess.run(args, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-main", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--seeds-other", type=int, nargs="+", default=[42])
    parser.add_argument("--train-cfg", type=Path, default=ROOT / "configs" / "train" / "full.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "E1_summary.csv")
    args = parser.parse_args()

    ensure_dir(args.out.parent)

    rows: list[dict] = []
    py = sys.executable

    all_models = list(DEEP_MODELS) + list(CLASSICAL_MODELS)
    for name in all_models:
        seeds = args.seeds_main if name in {"cnn_lstm", "xgboost", "random_forest", "logistic_regression"} else args.seeds_other
        model_cfg = ROOT / "configs" / "models" / f"{name}.yaml"
        if not model_cfg.exists():
            LOG.warning("No config for %s — skipping", name)
            continue
        f1s, pr_aucs, fprs, recalls, precisions = [], [], [], [], []
        for seed in seeds:
            try:
                _run([py, str(ROOT / "scripts" / "03_train.py"), "--model", str(model_cfg), "--train", str(args.train_cfg), "--seed", str(seed)])
                suffix = "joblib" if name in CLASSICAL_MODELS else "pt"
                ckpt = ROOT / "models" / f"{name}_seed{seed}.{suffix}"
                _run([py, str(ROOT / "scripts" / "04_evaluate.py"), "--model", str(model_cfg), "--checkpoint", str(ckpt), "--seed", str(seed)])
                eval_path = ROOT / "experiments" / "runs" / f"{name}_seed{seed}_eval.json"
                if eval_path.exists():
                    payload = __import__("json").loads(eval_path.read_text(encoding="utf-8"))
                    t = payload["test"]
                    f1s.append(t["f1"]); pr_aucs.append(t["pr_auc"]); fprs.append(t["fpr"])
                    recalls.append(t["recall"]); precisions.append(t["precision"])
            except subprocess.CalledProcessError as exc:
                LOG.error("Run failed for %s seed=%d: %s", name, seed, exc)
                continue

        if not f1s:
            continue
        rows.append(
            {
                "model": name,
                "seeds": len(f1s),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=0)) if len(f1s) > 1 else 0.0,
                "pr_auc_mean": float(np.mean(pr_aucs)),
                "fpr_mean": float(np.mean(fprs)),
                "recall_mean": float(np.mean(recalls)),
                "precision_mean": float(np.mean(precisions)),
            }
        )

    df = pd.DataFrame(rows).sort_values("f1_mean", ascending=False)
    df.to_csv(args.out, index=False)
    LOG.info("E1 summary -> %s", args.out)
    LOG.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
