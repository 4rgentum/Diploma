"""Fit the Preprocessor on UNSW-NB15 train, build windows, save artefacts.

Outputs (under ``data/processed/``):
    preprocessor.json     — fitted preprocessor (load via Preprocessor.load).
    train_windows.npz     — windows + labels + attack_cat for the training set.
    val_windows.npz       — same for the validation hold-out.
    test_windows.npz      — same for the official test set.
    metadata.json         — dataset statistics + paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.data import (  # noqa: E402
    Preprocessor,
    WindowBuilder,
    load_unsw_nb15,
    train_val_split,
)
from diploma_nids.utils import dump_json, ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.02_build_dataset")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data" / "unsw_nb15.yaml")
    parser.add_argument("--preproc", type=Path, default=ROOT / "configs" / "preprocess" / "default.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--smoke", action="store_true", help="Use 5% of the data for a fast smoke test.")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    pp_cfg = load_yaml(args.preproc)
    out = ensure_dir(args.out)

    data_dir = ROOT / data_cfg["data_dir"]
    LOG.info("Loading UNSW-NB15 from %s", data_dir)
    train_df, test_df = load_unsw_nb15(
        data_dir, train_name=data_cfg["train_csv"], test_name=data_cfg["test_csv"]
    )

    if args.smoke:
        train_df = train_df.sample(frac=0.05, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=0.05, random_state=42).reset_index(drop=True)
        LOG.warning("[smoke] using 5%% of the data — for pipeline verification only")

    LOG.info("Train rows=%d, test rows=%d", len(train_df), len(test_df))

    LOG.info("Splitting train -> train/val (val_fraction=%.2f)", data_cfg["val_fraction"])
    tr_df, val_df = train_val_split(
        train_df,
        val_fraction=float(data_cfg["val_fraction"]),
        seed=int(data_cfg.get("val_seed", 42)),
    )

    LOG.info("Fitting preprocessor on train (rows=%d)", len(tr_df))
    pp = Preprocessor(
        categorical_onehot=pp_cfg.get("categorical_onehot"),
        categorical_freq=pp_cfg.get("categorical_freq"),
        clip_low_q=pp_cfg.get("clip_low_q", 0.001),
        clip_high_q=pp_cfg.get("clip_high_q", 0.999),
    )
    pp.fit(tr_df)
    pp.save(out / "preprocessor.json")
    LOG.info("Preprocessor saved (output_dim=%d) -> %s", pp.output_dim, out / "preprocessor.json")

    wb = WindowBuilder(
        window=int(pp_cfg.get("window", 32)),
        stride=int(pp_cfg.get("stride", 8)),
        agg=str(pp_cfg.get("agg", "last")),
    )

    for name, df in (("train", tr_df), ("val", val_df), ("test", test_df)):
        X = pp.transform(df)
        y = df["label"].to_numpy().astype(np.int64)
        windows, labels = wb.build(X, y)
        # Aligned per-row attack_cat for the windows — we take the cat of the
        # last flow in each window (matches the 'last' label aggregation).
        cats = df["attack_cat"].astype(str).to_numpy()
        end_indices = np.arange(wb.window - 1, len(df), wb.stride)[: len(windows)]
        win_cat = cats[end_indices]
        path = out / f"{name}_windows.npz"
        np.savez_compressed(path, windows=windows, labels=labels, attack_cat=win_cat)
        LOG.info("%s: windows=%s, attack_share=%.3f -> %s", name, windows.shape, float(labels.mean()), path)

    meta = {
        "n_train_rows": int(len(tr_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "input_dim": pp.output_dim,
        "window": wb.window,
        "stride": wb.stride,
        "agg": wb.agg,
        "smoke": args.smoke,
    }
    dump_json(meta, out / "metadata.json")
    LOG.info("Done. Metadata -> %s", out / "metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
