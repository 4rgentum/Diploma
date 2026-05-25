"""Dataset loaders.

UNSW-NB15: official train/test split (Moustafa & Slay, 2015), CSVs named
``UNSW_NB15_training-set.csv`` / ``UNSW_NB15_testing-set.csv``.

CICIDS2017: CICFlowMeter output, multiple per-day CSVs in a folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    CICIDS_LABEL_COL,
    CICIDS_TO_UNSW_MAP,
    UNSW_ATTACK_CAT_COL,
    UNSW_BINARY,
    UNSW_CATEGORICAL,
    UNSW_FEATURES,
    UNSW_LABEL_COL,
    UNSW_NUMERIC,
)


def _coerce_unsw_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to expected dtypes, fill obvious sentinels."""
    df = df.copy()
    # Original UNSW CSVs sometimes contain " " or "-" in categorical cells.
    for col in UNSW_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({"": "unknown", "-": "unknown"})
    for col in UNSW_BINARY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    for col in UNSW_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
    if UNSW_LABEL_COL in df.columns:
        df[UNSW_LABEL_COL] = pd.to_numeric(df[UNSW_LABEL_COL], errors="coerce").fillna(0).astype(np.int8)
    if UNSW_ATTACK_CAT_COL in df.columns:
        df[UNSW_ATTACK_CAT_COL] = df[UNSW_ATTACK_CAT_COL].astype(str).str.strip()
        df.loc[df[UNSW_ATTACK_CAT_COL].isin(["", "-", "nan"]), UNSW_ATTACK_CAT_COL] = "Normal"
    return df


def load_unsw_nb15(
    data_dir: str | Path,
    *,
    train_name: str = "UNSW_NB15_training-set.csv",
    test_name: str = "UNSW_NB15_testing-set.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)``.

    The CSVs are read as-is; the function only ensures consistent dtypes and
    canonicalised categorical values. Splitting and pre-processing happen
    further down the pipeline.
    """
    root = Path(data_dir)
    train_path = root / train_name
    test_path = root / test_name
    if not train_path.exists():
        raise FileNotFoundError(f"UNSW-NB15 training CSV not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"UNSW-NB15 testing CSV not found at {test_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    # Some distributions of UNSW-NB15 ship an ``id`` column we don't need.
    for df in (train_df, test_df):
        df.drop(columns=[c for c in ("id", "Id") if c in df.columns], inplace=True, errors="ignore")

    required = set(UNSW_FEATURES) | {UNSW_LABEL_COL}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f"UNSW-NB15 training set is missing required columns: {sorted(missing)}")

    train_df = _coerce_unsw_dtypes(train_df)
    test_df = _coerce_unsw_dtypes(test_df)
    return train_df, test_df


def load_cicids2017(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    rename_to_unsw: bool = True,
) -> pd.DataFrame:
    """Read all CICIDS2017 CSVs into a single DataFrame.

    The default flow renames CICFlowMeter columns to UNSW-like names where a
    direct semantic mapping exists (see ``CICIDS_TO_UNSW_MAP``). The
    cross-evaluation script restricts to the common subset; here we just
    surface the data with a binary label.
    """
    root = Path(data_dir)
    csv_paths = sorted(root.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {pattern!r} in {root}")

    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        # Some CICIDS CSVs ship a stray BOM or leading whitespace in column names.
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        if CICIDS_LABEL_COL not in df.columns:
            raise ValueError(f"CICIDS CSV {p.name} has no '{CICIDS_LABEL_COL}' column")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, copy=False)

    if rename_to_unsw:
        df = df.rename(columns=CICIDS_TO_UNSW_MAP)

    # Binary label: 1 = attack, 0 = benign.
    raw_labels = df[CICIDS_LABEL_COL].astype(str).str.strip().str.lower()
    df[UNSW_LABEL_COL] = (~raw_labels.eq("benign")).astype(np.int8)
    df[UNSW_ATTACK_CAT_COL] = df[CICIDS_LABEL_COL].astype(str).str.strip()
    df.loc[df[UNSW_LABEL_COL] == 0, UNSW_ATTACK_CAT_COL] = "Normal"
    df.drop(columns=[CICIDS_LABEL_COL], inplace=True)

    # Replace +/-inf with NaN, then 0 — typical CICIDS artefact.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
