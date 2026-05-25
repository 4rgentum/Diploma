from __future__ import annotations

import numpy as np
import pandas as pd

from diploma_nids.data import temporal_split, train_val_split


def test_random_split_disjoint_and_complete() -> None:
    df = pd.DataFrame({"label": np.tile([0, 1], 100), "uid": np.arange(200)})
    tr, val = train_val_split(df, val_fraction=0.25, seed=0)
    assert len(tr) + len(val) == len(df)
    # The split is reset_index-ed; use the uid column to verify disjointness.
    assert set(tr["uid"]).isdisjoint(set(val["uid"]))
    assert set(tr["uid"]) | set(val["uid"]) == set(range(200))


def test_random_split_is_reproducible() -> None:
    df = pd.DataFrame({"label": np.tile([0, 1], 100)})
    tr1, val1 = train_val_split(df, val_fraction=0.2, seed=42)
    tr2, val2 = train_val_split(df, val_fraction=0.2, seed=42)
    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(val1, val2)


def test_temporal_split_keeps_order() -> None:
    df = pd.DataFrame({"label": np.arange(100)})
    tr, val = temporal_split(df, val_fraction=0.2)
    assert tr["label"].iloc[0] == 0
    assert val["label"].iloc[0] == int(0.8 * 100)
