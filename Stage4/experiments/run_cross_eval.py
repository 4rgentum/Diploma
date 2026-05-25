"""Cross-evaluation UNSW → CICIDS2017.

Train on UNSW, evaluate on CICIDS2017 using the common-feature subset.
Restricted to the 13 semantically-mappable features (Stage 2 Table for
CICIDS↔UNSW). We rebuild a minimal preprocessor on the common features
and re-train CNN-LSTM, then score the CICIDS test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "Stage3"
STAGE4 = ROOT / "Stage4"
sys.path.insert(0, str(STAGE3 / "src"))

from diploma_nids.data import Preprocessor, WindowBuilder, load_unsw_nb15  # noqa: E402
from diploma_nids.data.load import load_cicids2017  # noqa: E402
from diploma_nids.eval import binary_metrics, find_threshold_for_f1_max  # noqa: E402
from diploma_nids.models import build_model  # noqa: E402
from diploma_nids.training import TrainerConfig, train_deep  # noqa: E402
from diploma_nids.utils import dump_json, ensure_dir, get_logger, set_seed  # noqa: E402

LOG = get_logger("Stage4.cross_eval")

# Common-feature subset (must exist in both UNSW and CICIDS after CICIDS rename).
COMMON_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
]


def main() -> int:
    set_seed(42)
    out_tbl = ensure_dir(STAGE4 / "results" / "tables")

    # 1. Load UNSW train, restrict to common features
    LOG.info("Loading UNSW…")
    unsw_train, unsw_test = load_unsw_nb15(STAGE3 / "data" / "unsw_nb15")
    unsw_train = unsw_train[COMMON_FEATURES + ["label", "attack_cat"]].copy()
    unsw_test = unsw_test[COMMON_FEATURES + ["label", "attack_cat"]].copy()

    # 2. Load CICIDS, restrict to common features
    LOG.info("Loading CICIDS2017…")
    cic = load_cicids2017(STAGE3 / "data" / "cicids2017")
    keep = [c for c in COMMON_FEATURES if c in cic.columns] + ["label", "attack_cat"]
    cic = cic[keep].copy()
    # Some common features may be absent (CICIDS schema slightly differs); fill with 0.
    for col in COMMON_FEATURES:
        if col not in cic.columns:
            cic[col] = 0.0

    LOG.info("Subsampling CICIDS for fair comparison (200k rows)…")
    cic = cic.sample(min(200_000, len(cic)), random_state=42).reset_index(drop=True)

    # 3. Fit a minimal preprocessor on UNSW common features
    pp = Preprocessor(
        numeric_cols=COMMON_FEATURES,
        binary_cols=[],
        categorical_onehot=[],
        categorical_freq=[],
        log_transform=COMMON_FEATURES,
    ).fit(unsw_train)
    LOG.info("Preprocessor fitted (output_dim=%d)", pp.output_dim)

    wb = WindowBuilder(window=32, stride=8, agg="last")

    X_tr = pp.transform(unsw_train)
    X_te_unsw = pp.transform(unsw_test)
    X_te_cic = pp.transform(cic)

    Xw_tr, yw_tr = wb.build(X_tr, unsw_train["label"].to_numpy().astype(np.int64))
    Xw_te_unsw, yw_te_unsw = wb.build(X_te_unsw, unsw_test["label"].to_numpy().astype(np.int64))
    Xw_te_cic, yw_te_cic = wb.build(X_te_cic, cic["label"].to_numpy().astype(np.int64))

    LOG.info("UNSW train windows: %s, UNSW test windows: %s, CICIDS test windows: %s",
             Xw_tr.shape, Xw_te_unsw.shape, Xw_te_cic.shape)

    # 4. Train CNN-LSTM on common features
    cfg = {
        "name": "cnn_lstm",
        "input_dim": int(Xw_tr.shape[-1]),
        "window": 32,
        "params": {
            "cnn_channels": [64, 128],
            "kernel_size": 3,
            "lstm_hidden": 96,
            "bidirectional": True,
            "head_hidden": 64,
            "dropout": 0.25,
            "use_attention_pool": True,
        },
    }
    model = build_model(cfg)
    LOG.info("Training CNN-LSTM on common features…")
    trainer_cfg = TrainerConfig(epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4)
    train_deep(model, Xw_tr, yw_tr, Xw_te_unsw[: len(Xw_te_unsw) // 2], yw_te_unsw[: len(yw_te_unsw) // 2], cfg=trainer_cfg)

    # 5. Eval on both
    p_unsw = model.predict_proba(Xw_te_unsw)
    tau_unsw = find_threshold_for_f1_max(yw_te_unsw, p_unsw)
    m_unsw = binary_metrics(yw_te_unsw, p_unsw, threshold=tau_unsw)

    p_cic = model.predict_proba(Xw_te_cic)
    tau_cic = find_threshold_for_f1_max(yw_te_cic, p_cic)
    m_cic_optimal = binary_metrics(yw_te_cic, p_cic, threshold=tau_cic)
    # And cross-applied: use UNSW threshold on CICIDS (the harsher real-world setting).
    m_cic_unsw_tau = binary_metrics(yw_te_cic, p_cic, threshold=tau_unsw)

    delta_f1 = m_unsw["f1"] - m_cic_unsw_tau["f1"]

    payload = {
        "common_features": COMMON_FEATURES,
        "n_unsw_train_windows": int(Xw_tr.shape[0]),
        "n_unsw_test_windows": int(Xw_te_unsw.shape[0]),
        "n_cicids_test_windows": int(Xw_te_cic.shape[0]),
        "unsw_f1_max": m_unsw["f1"],
        "unsw_pr_auc": m_unsw["pr_auc"],
        "unsw_threshold": tau_unsw,
        "cicids_f1_max": m_cic_optimal["f1"],
        "cicids_pr_auc": m_cic_optimal["pr_auc"],
        "cicids_threshold_optimal": tau_cic,
        "cicids_f1_using_unsw_threshold": m_cic_unsw_tau["f1"],
        "delta_f1_transfer": delta_f1,
        "nfr7_threshold": 0.20,
        "nfr7_met": delta_f1 <= 0.20,
    }
    dump_json(payload, out_tbl / "cross_eval_summary.json")
    pd.DataFrame([payload]).to_csv(out_tbl / "cross_eval_summary.csv", index=False)
    LOG.info("Cross-eval summary → %s", out_tbl / "cross_eval_summary.json")
    LOG.info("UNSW F1=%.4f → CICIDS F1=%.4f (Δ=%.4f, NFR-7 met: %s)",
             m_unsw["f1"], m_cic_unsw_tau["f1"], delta_f1, payload["nfr7_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
