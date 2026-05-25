"""Preprocessor — sklearn-compatible ``fit/transform`` with full JSON serialisation.

Design points (Stage 2 §2):

* All statistics (median, IQR, frequency tables, quantile clips) are learned
  **only on the training set**. The same fitted transformer is applied to
  val/test/stream — no test-time leakage.
* Heavy-tailed numeric features go through ``log1p`` before robust scaling
  (median / IQR). Sentinel-zero values become ``log1p(0) = 0`` cleanly.
* Categorical features split by cardinality:
    - ``proto``, ``state`` → one-hot;
    - ``service`` → frequency encoding (training-set rel-frequency, unknowns
      get 0.0).
* The fitted state is a plain JSON dict, so the same artefact runs in train,
  eval, and the online service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    UNSW_BINARY,
    UNSW_CATEGORICAL,
    UNSW_LOG_TRANSFORM,
    UNSW_NUMERIC,
)


@dataclass
class PreprocessorState:
    """Serialisable fitted state."""

    numeric_cols: list[str] = field(default_factory=list)
    binary_cols: list[str] = field(default_factory=list)
    categorical_onehot: list[str] = field(default_factory=list)
    categorical_freq: list[str] = field(default_factory=list)
    log_transform: list[str] = field(default_factory=list)

    quantile_low: dict[str, float] = field(default_factory=dict)
    quantile_high: dict[str, float] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)
    iqr: dict[str, float] = field(default_factory=dict)

    onehot_categories: dict[str, list[str]] = field(default_factory=dict)
    freq_table: dict[str, dict[str, float]] = field(default_factory=dict)

    output_columns: list[str] = field(default_factory=list)


class Preprocessor:
    """Robust, declarative preprocessor for UNSW-NB15-shaped flow data."""

    def __init__(
        self,
        *,
        numeric_cols: list[str] | None = None,
        binary_cols: list[str] | None = None,
        categorical_onehot: list[str] | None = None,
        categorical_freq: list[str] | None = None,
        log_transform: list[str] | None = None,
        clip_low_q: float = 0.001,
        clip_high_q: float = 0.999,
    ) -> None:
        self.numeric_cols = list(numeric_cols if numeric_cols is not None else UNSW_NUMERIC)
        self.binary_cols = list(binary_cols if binary_cols is not None else UNSW_BINARY)
        self.categorical_onehot = list(
            categorical_onehot if categorical_onehot is not None else ["proto", "state"]
        )
        self.categorical_freq = list(
            categorical_freq if categorical_freq is not None else ["service"]
        )
        self.log_transform = list(
            log_transform if log_transform is not None else UNSW_LOG_TRANSFORM
        )
        # Intersect log_transform with numeric to avoid configuration drift.
        self.log_transform = [c for c in self.log_transform if c in self.numeric_cols]

        self.clip_low_q = float(clip_low_q)
        self.clip_high_q = float(clip_high_q)
        self.state = PreprocessorState()
        self._fitted = False

    # ------------------------------------------------------------------ #
    # fit / transform                                                     #
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = self._canonicalise_categoricals(df)

        self.state = PreprocessorState(
            numeric_cols=list(self.numeric_cols),
            binary_cols=list(self.binary_cols),
            categorical_onehot=list(self.categorical_onehot),
            categorical_freq=list(self.categorical_freq),
            log_transform=list(self.log_transform),
        )

        # Quantile clips on raw numeric (pre-log) — symmetric tail trimming.
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            ql, qh = df[col].quantile([self.clip_low_q, self.clip_high_q])
            self.state.quantile_low[col] = float(ql)
            self.state.quantile_high[col] = float(qh)

        # Robust-scaler stats on post-log values.
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            v = df[col].clip(self.state.quantile_low[col], self.state.quantile_high[col]).astype(np.float64)
            if col in self.log_transform:
                v = np.log1p(np.clip(v, a_min=0.0, a_max=None))
            med = float(v.median())
            q25 = float(v.quantile(0.25))
            q75 = float(v.quantile(0.75))
            iqr = q75 - q25
            if iqr < 1e-9:
                iqr = 1.0  # constant column — leave as zero-mean
            self.state.median[col] = med
            self.state.iqr[col] = iqr

        # One-hot categories — fix the order so train/eval/stream agree.
        for col in self.categorical_onehot:
            if col not in df.columns:
                continue
            cats = sorted(str(x) for x in df[col].dropna().unique())
            if "unknown" not in cats:
                cats.append("unknown")
            self.state.onehot_categories[col] = cats

        # Frequency encoding — relative training-set frequencies.
        n = len(df)
        for col in self.categorical_freq:
            if col not in df.columns:
                continue
            counts = df[col].value_counts(dropna=False)
            self.state.freq_table[col] = {str(k): float(v) / float(n) for k, v in counts.items()}

        self.state.output_columns = self._compute_output_columns()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor.fit must be called before transform.")

        df = self._canonicalise_categoricals(df)
        parts: list[np.ndarray] = []

        # Numeric
        for col in self.state.numeric_cols:
            if col in df.columns:
                v = df[col].astype(np.float64)
            else:
                v = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
            ql = self.state.quantile_low.get(col, float(v.min()))
            qh = self.state.quantile_high.get(col, float(v.max()))
            v = v.clip(ql, qh)
            if col in self.state.log_transform:
                v = np.log1p(np.clip(v.to_numpy(), a_min=0.0, a_max=None))
            else:
                v = v.to_numpy()
            med = self.state.median.get(col, 0.0)
            iqr = self.state.iqr.get(col, 1.0)
            v = (v - med) / iqr
            parts.append(v.astype(np.float32).reshape(-1, 1))

        # Binary
        for col in self.state.binary_cols:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32).to_numpy()
            else:
                v = np.zeros(len(df), dtype=np.float32)
            parts.append(v.reshape(-1, 1))

        # Categorical — one-hot
        for col in self.state.categorical_onehot:
            cats = self.state.onehot_categories.get(col, ["unknown"])
            if col in df.columns:
                v = df[col].astype(str).to_numpy()
            else:
                v = np.full(len(df), "unknown", dtype=object)
            ohe = np.zeros((len(v), len(cats)), dtype=np.float32)
            cat_idx = {c: i for i, c in enumerate(cats)}
            unknown_idx = cat_idx.get("unknown")
            for row, val in enumerate(v):
                idx = cat_idx.get(val, unknown_idx)
                if idx is not None:
                    ohe[row, idx] = 1.0
            parts.append(ohe)

        # Categorical — frequency
        for col in self.state.categorical_freq:
            tbl = self.state.freq_table.get(col, {})
            if col in df.columns:
                v = df[col].astype(str).to_numpy()
            else:
                v = np.full(len(df), "unknown", dtype=object)
            arr = np.array([tbl.get(x, 0.0) for x in v], dtype=np.float32)
            parts.append(arr.reshape(-1, 1))

        X = np.concatenate(parts, axis=1)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    # serialisation                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save un-fitted preprocessor.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clip_low_q": self.clip_low_q,
            "clip_high_q": self.clip_high_q,
            "state": {
                "numeric_cols": self.state.numeric_cols,
                "binary_cols": self.state.binary_cols,
                "categorical_onehot": self.state.categorical_onehot,
                "categorical_freq": self.state.categorical_freq,
                "log_transform": self.state.log_transform,
                "quantile_low": self.state.quantile_low,
                "quantile_high": self.state.quantile_high,
                "median": self.state.median,
                "iqr": self.state.iqr,
                "onehot_categories": self.state.onehot_categories,
                "freq_table": self.state.freq_table,
                "output_columns": self.state.output_columns,
            },
        }
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        with Path(path).open("r", encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)
        st = payload["state"]
        pp = cls(
            numeric_cols=st["numeric_cols"],
            binary_cols=st["binary_cols"],
            categorical_onehot=st["categorical_onehot"],
            categorical_freq=st["categorical_freq"],
            log_transform=st["log_transform"],
            clip_low_q=payload.get("clip_low_q", 0.001),
            clip_high_q=payload.get("clip_high_q", 0.999),
        )
        pp.state = PreprocessorState(
            numeric_cols=st["numeric_cols"],
            binary_cols=st["binary_cols"],
            categorical_onehot=st["categorical_onehot"],
            categorical_freq=st["categorical_freq"],
            log_transform=st["log_transform"],
            quantile_low=st["quantile_low"],
            quantile_high=st["quantile_high"],
            median=st["median"],
            iqr=st["iqr"],
            onehot_categories=st["onehot_categories"],
            freq_table=st["freq_table"],
            output_columns=st["output_columns"],
        )
        pp._fitted = True
        return pp

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #
    def _canonicalise_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in (*self.categorical_onehot, *self.categorical_freq):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().replace({"": "unknown", "-": "unknown"})
        return df

    def _compute_output_columns(self) -> list[str]:
        cols: list[str] = []
        cols.extend(self.numeric_cols)
        cols.extend(self.binary_cols)
        for col in self.categorical_onehot:
            for cat in self.state.onehot_categories.get(col, []):
                cols.append(f"{col}={cat}")
        for col in self.categorical_freq:
            cols.append(f"{col}_freq")
        return cols

    @property
    def output_dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return len(self.state.output_columns)
