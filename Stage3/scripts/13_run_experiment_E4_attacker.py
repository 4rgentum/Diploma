"""Experiment E4 — stress test with the FSM agent. Per-state recall.

Uses empirical templates fitted on UNSW-NB15 train (fixes the OOD problem of
the previous prototype where PSI was already 0.31 at intensity 0).
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

from diploma_nids.attacker import runtime_from_config  # noqa: E402
from diploma_nids.data import Preprocessor, load_unsw_nb15  # noqa: E402
from diploma_nids.eval import binary_metrics  # noqa: E402
from diploma_nids.inference import WindowScorer  # noqa: E402
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.utils import dump_json, ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.13_E4")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "configs" / "models" / "cnn_lstm.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "cnn_lstm_seed42.pt")
    parser.add_argument("--eval-json", type=Path, default=ROOT / "experiments" / "runs" / "cnn_lstm_seed42_eval.json")
    parser.add_argument("--preprocessor", type=Path, default=ROOT / "data" / "processed" / "preprocessor.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "attacker" / "policy.yaml")
    parser.add_argument("--ticks", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-tbl", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()

    ensure_dir(args.out_tbl)
    ensure_dir(args.out_fig)

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])

    pp = Preprocessor.load(args.preprocessor)
    model_cfg = load_yaml(args.model)
    model_cfg.setdefault("input_dim", pp.output_dim)
    model_cfg.setdefault("window", 32)
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu").get("state_dict"))
    model.eval()

    data_cfg = load_yaml(ROOT / "configs" / "data" / "unsw_nb15.yaml")
    train_df, _ = load_unsw_nb15(
        ROOT / data_cfg["data_dir"], train_name=data_cfg["train_csv"], test_name=data_cfg["test_csv"]
    )

    rt = runtime_from_config(args.policy, seed=args.seed, empirical_df=train_df)
    df = rt.collect_dataframe(args.ticks)

    scorer = WindowScorer(model=model, preprocessor=pp, window=32, stride=8)
    indices = []
    scores = []
    for end_idx, sc in scorer.score_batch(df):
        indices.append(end_idx)
        scores.append(sc)
    scores = np.asarray(scores)
    indices = np.asarray(indices)
    win_state = df["fsm_state"].to_numpy()[indices]
    win_label = df["label"].to_numpy()[indices].astype(np.int64)
    y_pred = (scores >= threshold).astype(np.int64)

    metrics = binary_metrics(win_label, scores, threshold=threshold)
    metrics["windows"] = int(len(scores))
    metrics["attack_windows"] = int(win_label.sum())
    dump_json(metrics, args.out_tbl / "E4_metrics.json")

    # Per-state recall.
    states = sorted(set(win_state.tolist()))
    rows = []
    for st in states:
        mask = win_state == st
        if not mask.any():
            continue
        if st == "NORMAL" or st.startswith("DRIFT_"):
            fpr = float(y_pred[mask].mean())
            rows.append({"state": st, "support": int(mask.sum()), "value": fpr, "type": "fpr"})
        else:
            recall = float(y_pred[mask & (win_label == 1)].mean()) if (win_label[mask] == 1).any() else 0.0
            rows.append({"state": st, "support": int(mask.sum()), "value": recall, "type": "recall"})
    pc = pd.DataFrame(rows).sort_values(["type", "state"])
    pc.to_csv(args.out_tbl / "E4_per_state_recall.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    color_map = {"recall": "steelblue", "fpr": "salmon"}
    ax.barh(pc["state"], pc["value"], color=[color_map[t] for t in pc["type"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall / FPR")
    ax.set_title("E4: per-state behaviour (CNN-LSTM vs FSM agent)")
    fig.tight_layout()
    fig.savefig(args.out_fig / "E4_per_state_recall.png", dpi=140)
    fig.savefig(args.out_fig / "E4_per_state_recall.pdf")

    LOG.info("E4 done: F1=%.3f Recall=%.3f FPR=%.3f", metrics["f1"], metrics["recall"], metrics["fpr"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
