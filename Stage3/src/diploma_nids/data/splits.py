"""Train / val / test splitting strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def train_val_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
    stratify_col: str | None = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random stratified split of a DataFrame into train / val.

    Stratification is applied only if ``stratify_col`` is present.
    """
    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError("val_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)

    if stratify_col and stratify_col in df.columns:
        idx_parts: list[np.ndarray] = []
        for cls in df[stratify_col].unique():
            # ``to_numpy`` on pandas 3.x returns a read-only view; copy first.
            cls_idx = np.array(df.index[df[stratify_col] == cls].to_numpy(), copy=True)
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * val_fraction)))
            idx_parts.append(cls_idx[:n_val])
        val_idx = np.concatenate(idx_parts)
    else:
        all_idx = np.array(df.index.to_numpy(), copy=True)
        rng.shuffle(all_idx)
        n_val = int(round(len(all_idx) * val_fraction))
        val_idx = all_idx[:n_val]

    val_idx_set = set(val_idx.tolist())
    val_mask = df.index.isin(val_idx_set)

    val_df = df.loc[val_mask].reset_index(drop=True)
    train_df = df.loc[~val_mask].reset_index(drop=True)
    return train_df, val_df


def temporal_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split — preserves the natural order of the rows.

    Useful when the dataframe is already chronologically sorted (e.g. a stream
    capture). For UNSW-NB15 the official CSVs are not strictly chronological,
    so ``train_val_split`` is the default.
    """
    n = len(df)
    if n < 2:
        raise ValueError("temporal_split requires at least 2 rows")
    n_train = int(round(n * (1 - val_fraction)))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:].reset_index(drop=True)
    return train_df, val_df
