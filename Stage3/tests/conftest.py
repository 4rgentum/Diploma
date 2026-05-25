"""Shared fixtures + sys.path bootstrap (so ``pytest`` works from repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def toy_unsw_df() -> pd.DataFrame:
    """A tiny but schema-compatible UNSW-NB15 frame for unit tests."""
    from diploma_nids.data.schema import UNSW_BINARY, UNSW_CATEGORICAL, UNSW_NUMERIC

    n = 256
    rng = np.random.default_rng(123)
    data: dict[str, np.ndarray] = {}
    for col in UNSW_NUMERIC:
        data[col] = rng.exponential(scale=2.0, size=n).astype(np.float32)
    for col in UNSW_CATEGORICAL:
        data[col] = rng.choice(["tcp", "udp", "fin", "http", "-"], size=n)
    for col in UNSW_BINARY:
        data[col] = rng.integers(0, 2, size=n).astype(np.int8)

    df = pd.DataFrame(data)
    df["label"] = rng.integers(0, 2, size=n).astype(np.int8)
    df["attack_cat"] = np.where(df["label"] == 1, "DoS", "Normal")
    return df
