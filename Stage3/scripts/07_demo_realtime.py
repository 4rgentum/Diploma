"""Realtime demo: stream the FSM agent into a Detector → AlertFormer pipeline,
plot the timeline, dump alerts/drift jsonl. Mirrors Stage 2 §8.8.
"""

from __future__ import annotations

import argparse
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
from diploma_nids.eval import drift_report  # noqa: E402
from diploma_nids.inference import AlertFormer, WindowScorer  # noqa: E402
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.utils import dump_json, ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.07_demo")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=ROOT / "configs" / "pipeline" / "realtime.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs" / "models" / "cnn_lstm.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "cnn_lstm_seed42.pt")
    parser.add_argument("--preprocessor", type=Path, default=ROOT / "data" / "processed" / "preprocessor.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "demo")
    parser.add_argument("--ticks", type=int, default=300)
    args = parser.parse_args()

    cfg = load_yaml(args.pipeline)
    ensure_dir(args.out)
    pp = Preprocessor.load(args.preprocessor)

    model_cfg = load_yaml(args.model)
    model_cfg.setdefault("input_dim", pp.output_dim)
    model_cfg.setdefault("window", int(cfg["scorer"]["window"]))
    model = build_model(model_cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    scorer = WindowScorer(
        model=model,
        preprocessor=pp,
        window=int(cfg["scorer"]["window"]),
        stride=int(cfg["scorer"]["stride"]),
    )
    alert_former = AlertFormer(
        threshold=args.threshold,
        dedup_seconds=float(cfg["alerting"]["dedup_seconds"]),
        history_size=int(cfg["alerting"]["history_size"]),
        model_version="cnn_lstm_demo",
    )

    # Use empirical templates if dataset is available — otherwise parametric.
    data_cfg = load_yaml(ROOT / "configs" / "data" / "unsw_nb15.yaml")
    df_unsw = None
    try:
        train_df, _ = load_unsw_nb15(
            ROOT / data_cfg["data_dir"], train_name=data_cfg["train_csv"], test_name=data_cfg["test_csv"]
        )
        df_unsw = train_df
    except FileNotFoundError:
        LOG.warning("UNSW data not available — using parametric templates")

    rt = runtime_from_config(cfg["agent"]["policy"], seed=int(cfg["agent"]["seed"]), empirical_df=df_unsw)

    score_log: list[tuple[int, float, str]] = []
    alerts: list[dict] = []

    for i, tick in enumerate(rt.stream(args.ticks)):
        score = scorer.push(tick.flow)
        if score is None:
            continue
        score_log.append((i, score, tick.fsm_state))
        a = alert_former.maybe_emit(
            ts=float(i),
            score=score,
            attack_cat=tick.attack_cat,
            ground_truth={"fsm_state": tick.fsm_state, "drift_type": tick.drift_type},
        )
        if a is not None:
            alerts.append(a.to_dict())

    # Persist alert log + summary.
    alerts_path = args.out / "demo_alerts.jsonl"
    with alerts_path.open("w", encoding="utf-8") as fh:
        for a in alerts:
            fh.write(__import__("json").dumps(a, ensure_ascii=False) + "\n")
    LOG.info("Alerts: %d -> %s", len(alerts), alerts_path)

    summary = {
        "n_alerts": len(alerts),
        "n_scored_windows": len(score_log),
        "n_ticks": args.ticks,
        "threshold": args.threshold,
        "severity_breakdown": {
            sev: sum(1 for a in alerts if a["severity"] == sev)
            for sev in ("info", "low", "medium", "high", "critical")
        },
    }
    dump_json(summary, args.out / "demo_summary.json")

    # Drift sweep on the streamed flows (chunked).
    drift_records = []
    if df_unsw is not None:
        ref = pp.transform(df_unsw.sample(2000, random_state=0))
        # Reuse the flows from the run for the current window.
        # For simplicity, sample synthetic equivalents.
        from diploma_nids.attacker import runtime_from_config as _rt
        rt2 = _rt(cfg["agent"]["policy"], seed=99, empirical_df=df_unsw)
        cur_df = rt2.collect_dataframe(2000)
        cur = pp.transform(cur_df)
        drift_records.append(drift_report(ref, cur))
        with (args.out / "demo_drift.jsonl").open("w", encoding="utf-8") as fh:
            for d in drift_records:
                fh.write(__import__("json").dumps(d, ensure_ascii=False) + "\n")

    # Timeline plot.
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ts = np.array([t for t, _, _ in score_log])
    sc = np.array([s for _, s, _ in score_log])
    ax[0].plot(ts, sc, lw=0.8, color="steelblue")
    ax[0].axhline(args.threshold, color="red", lw=0.8, linestyle="--", label=f"τ={args.threshold:.3f}")
    ax[0].set_ylabel("score")
    ax[0].set_title("CNN-LSTM live scores")
    ax[0].legend(loc="upper right")
    states = [st for _, _, st in score_log]
    state_names = sorted(set(states))
    state_idx = [state_names.index(s) for s in states]
    ax[1].scatter(ts, state_idx, s=4, c="darkorange")
    ax[1].set_yticks(range(len(state_names)))
    ax[1].set_yticklabels(state_names, fontsize=7)
    ax[1].set_xlabel("tick")
    ax[1].set_title("FSM state timeline")
    fig.tight_layout()
    fig.savefig(args.out / "demo_timeline.png", dpi=140)
    fig.savefig(args.out / "demo_timeline.pdf")
    LOG.info("Timeline -> %s", args.out / "demo_timeline.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
