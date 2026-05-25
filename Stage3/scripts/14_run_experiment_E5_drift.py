"""Experiment E5 — controlled-drift sweep.

For each (drift_type, intensity) pair, generate a fresh agent stream, push it
through the detector and the drift monitor, and record F1 / PSI / MMD.
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
from diploma_nids.attacker.drift_injector import DriftInjector, DriftType  # noqa: E402
from diploma_nids.data import Preprocessor, load_unsw_nb15  # noqa: E402
from diploma_nids.data.schema import UNSW_NUMERIC  # noqa: E402
from diploma_nids.eval import binary_metrics, drift_report  # noqa: E402
from diploma_nids.inference import WindowScorer  # noqa: E402
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.utils import ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.14_E5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "configs" / "models" / "cnn_lstm.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "cnn_lstm_seed42.pt")
    parser.add_argument("--preprocessor", type=Path, default=ROOT / "data" / "processed" / "preprocessor.json")
    parser.add_argument("--eval-json", type=Path, default=ROOT / "experiments" / "runs" / "cnn_lstm_seed42_eval.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "attacker" / "policy.yaml")
    parser.add_argument("--drift-cfg", type=Path, default=ROOT / "configs" / "attacker" / "drift.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "E5_drift_sweep.csv")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "results" / "figures" / "E5_drift_sweep.png")
    args = parser.parse_args()

    ensure_dir(args.out.parent)
    ensure_dir(args.out_fig.parent)

    pp = Preprocessor.load(args.preprocessor)
    model_cfg = load_yaml(args.model)
    model_cfg.setdefault("input_dim", pp.output_dim)
    model_cfg.setdefault("window", 32)
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu").get("state_dict"))
    model.eval()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])

    data_cfg = load_yaml(ROOT / "configs" / "data" / "unsw_nb15.yaml")
    train_df, _ = load_unsw_nb15(
        ROOT / data_cfg["data_dir"], train_name=data_cfg["train_csv"], test_name=data_cfg["test_csv"]
    )
    drift_cfg = load_yaml(args.drift_cfg)
    n_attack = int(drift_cfg["n_attack_samples"])
    n_normal = int(drift_cfg["n_normal_samples"])
    n_ticks = n_attack + n_normal

    # Reference window from UNSW normal flows — this is the canonical "no drift" baseline.
    ref_df = train_df[train_df["label"] == 0].sample(int(drift_cfg["reference_normal_samples"]), random_state=0)
    ref_features = pp.transform(ref_df)

    # Feature std dict for the injector.
    feat_std = {c: float(train_df[c].std()) if c in train_df.columns else 1.0 for c in UNSW_NUMERIC}

    rows = []
    for dt in drift_cfg["drift_types"]:
        for intensity in drift_cfg["intensities"]:
            rt = runtime_from_config(args.policy, seed=args.seed, empirical_df=train_df)
            rt.injector = DriftInjector(drift_type=DriftType(dt), intensity=float(intensity), feature_std=feat_std, seed=args.seed)
            df = rt.collect_dataframe(n_ticks)

            scorer = WindowScorer(model=model, preprocessor=pp, window=32, stride=8)
            scores = []
            indices = []
            for end_idx, sc in scorer.score_batch(df):
                indices.append(end_idx); scores.append(sc)
            scores = np.asarray(scores); indices = np.asarray(indices)
            y = df["label"].to_numpy().astype(np.int64)[indices]
            metrics = binary_metrics(y, scores, threshold=threshold)

            cur_features = pp.transform(df.head(2000))
            report = drift_report(ref_features, cur_features, psi_threshold=0.25)

            rows.append(
                {
                    "drift_type": dt,
                    "intensity": float(intensity),
                    "f1": metrics["f1"],
                    "pr_auc": metrics["pr_auc"],
                    "recall": metrics["recall"],
                    "fpr": metrics["fpr"],
                    "psi_mean": report["psi"]["mean"],
                    "psi_max": report["psi"]["max"],
                    "mmd": report["mmd"],
                    "drift_alarm": bool(report["drift_alarm"]),
                }
            )
            LOG.info("drift=%s intensity=%.2f F1=%.3f PSI=%.3f alarm=%s", dt, intensity, metrics["f1"], report["psi"]["mean"], report["drift_alarm"])

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for dt, g in df.groupby("drift_type"):
        ax[0].plot(g["intensity"], g["f1"], marker="o", label=f"{dt}")
        ax[1].plot(g["intensity"], g["psi_mean"], marker="o", label=f"{dt}")
    ax[1].axhline(0.25, color="red", linestyle="--", label="PSI=0.25")
    ax[0].set_xlabel("intensity"); ax[0].set_ylabel("F1"); ax[0].legend(); ax[0].set_title("F1 vs drift intensity")
    ax[1].set_xlabel("intensity"); ax[1].set_ylabel("mean PSI"); ax[1].legend(); ax[1].set_title("PSI vs drift intensity")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=140)
    fig.savefig(args.out_fig.with_suffix(".pdf"))
    LOG.info("E5 -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
