from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diploma_nids.data import Preprocessor


def test_fit_then_transform_no_nan(toy_unsw_df: pd.DataFrame) -> None:
    pp = Preprocessor().fit(toy_unsw_df)
    X = pp.transform(toy_unsw_df)
    assert X.dtype == np.float32
    assert not np.isnan(X).any()
    assert X.shape == (len(toy_unsw_df), pp.output_dim)


def test_save_load_roundtrip(tmp_path: Path, toy_unsw_df: pd.DataFrame) -> None:
    pp = Preprocessor().fit(toy_unsw_df)
    X1 = pp.transform(toy_unsw_df)
    pp.save(tmp_path / "pp.json")
    pp2 = Preprocessor.load(tmp_path / "pp.json")
    X2 = pp2.transform(toy_unsw_df)
    assert np.allclose(X1, X2)
    assert pp2.output_dim == pp.output_dim


def test_unknown_category_handled(toy_unsw_df: pd.DataFrame) -> None:
    pp = Preprocessor().fit(toy_unsw_df)
    weird = toy_unsw_df.head(3).copy()
    weird["proto"] = "totally_unseen"
    X = pp.transform(weird)
    assert X.shape == (3, pp.output_dim)


def test_transform_without_fit_raises(toy_unsw_df: pd.DataFrame) -> None:
    pp = Preprocessor()
    with pytest.raises(RuntimeError):
        pp.transform(toy_unsw_df)


def test_statistics_only_on_train(toy_unsw_df: pd.DataFrame) -> None:
    pp = Preprocessor().fit(toy_unsw_df.head(100))
    med1 = dict(pp.state.median)
    # Calling transform on extreme data should not mutate the fitted stats.
    extreme = toy_unsw_df.copy()
    extreme["sbytes"] = 1e9
    _ = pp.transform(extreme)
    assert pp.state.median == med1
