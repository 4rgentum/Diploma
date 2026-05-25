from __future__ import annotations

import numpy as np
import pytest

from diploma_nids.data import build_windows
from diploma_nids.data.windowing import WindowBuilder


def test_shapes_match_expected() -> None:
    X = np.arange(64 * 5, dtype=np.float32).reshape(64, 5)
    y = np.zeros(64, dtype=np.int64)
    win, lab = build_windows(X, y, window=16, stride=4)
    assert win.shape == ((64 - 16) // 4 + 1, 16, 5)
    assert lab.shape == (win.shape[0],)


def test_last_strategy_picks_last_flow_label() -> None:
    y = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    X = np.zeros((8, 3), dtype=np.float32)
    win, lab = build_windows(X, y, window=4, stride=2)
    assert win.shape[0] == lab.shape[0]
    assert lab[-1] == 1  # last window ends at index 7 with label 1


@pytest.mark.parametrize("agg", ["last", "majority", "any"])
def test_all_aggregations_run(agg: str) -> None:
    X = np.zeros((32, 3), dtype=np.float32)
    y = np.ones(32, dtype=np.int64)
    win, lab = build_windows(X, y, window=8, stride=4, agg=agg)
    assert win.shape == ((32 - 8) // 4 + 1, 8, 3)
    assert lab.shape[0] == win.shape[0]


def test_short_input_raises() -> None:
    X = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        build_windows(X, None, window=16, stride=4)


def test_builder_validates_args() -> None:
    with pytest.raises(ValueError):
        WindowBuilder(window=0, stride=1)
    with pytest.raises(ValueError):
        WindowBuilder(window=4, stride=0)
    with pytest.raises(ValueError):
        WindowBuilder(window=4, stride=2, agg="weird")  # type: ignore[arg-type]
