"""Experiment E2 — error analysis for the proposed CNN-LSTM model.

Builds the confusion matrix and per-attack-cat recall on the test set; saves
CSV and PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.eval import per_class_recall  # noqa: E402
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.utils import ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.11_E2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "configs" / "models" / "cnn_lstm.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "cnn_lstm_seed42.pt")
    parser.add_argument("--eval-json", type=Path, default=ROOT / "experiments" / "runs" / "cnn_lstm_seed42_eval.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--out-tbl", type=Path, default=ROOT / "results" / "tables")
    args = parser.parse_args()

    ensure_dir(args.out_fig)
    ensure_dir(args.out_tbl)

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    LOG.info("Using calibrated threshold τ=%.4f", threshold)

    test_npz = np.load(args.data / "test_windows.npz", allow_pickle=True)
    X_test, y_test = test_npz["windows"], test_npz["labels"]
    test_cat = test_npz["attack_cat"]

    model_cfg = load_yaml(args.model)
    model_cfg.setdefault("input_dim", int(X_test.shape[-1]))
    model_cfg.setdefault("window", int(X_test.shape[1]))
    model = build_model(model_cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    probs = model.predict_proba(X_test)
    y_pred = (probs >= threshold).astype(np.int64)

    # Confusion matrix.
    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    pd.DataFrame(cm, index=["normal_true", "attack_true"], columns=["normal_pred", "attack_pred"]).to_csv(
        args.out_tbl / "E2_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "attack"]); ax.set_yticklabels(["normal", "attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("CNN-LSTM — confusion matrix")
    fig.tight_layout()
    fig.savefig(args.out_fig / "E2_confusion_matrix.png", dpi=140)
    fig.savefig(args.out_fig / "E2_confusion_matrix.pdf")

    # Per-class recall.
    pc = per_class_recall(y_test, y_pred, test_cat)
    pc.to_csv(args.out_tbl / "E2_per_attack_recall.csv", index=False)

    attack_rows = pc[pc["type"] == "recall"].sort_values("recall_or_fpr")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(attack_rows["attack_cat"], attack_rows["recall_or_fpr"], color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall")
    ax.set_title("Per-attack recall (target FPR=0.01)")
    fig.tight_layout()
    fig.savefig(args.out_fig / "E2_per_attack_recall.png", dpi=140)
    fig.savefig(args.out_fig / "E2_per_attack_recall.pdf")

    LOG.info("E2 done. TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
