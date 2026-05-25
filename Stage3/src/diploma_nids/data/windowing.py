"""Sliding-window builder.

Given an ``(N, F)`` feature matrix and per-row labels, return an ``(M, W, F)``
tensor of overlapping windows and an ``(M,)`` label vector aggregated per the
selected strategy (last / majority / any). See Stage 2 §2.5.

The implementation is fully vectorised — no Python-level loops over rows —
which makes it usable in the streaming path as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AggStrategy = Literal["last", "majority", "any"]


@dataclass
class WindowBuilder:
    window: int = 32
    stride: int = 8
    agg: AggStrategy = "last"

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.agg not in ("last", "majority", "any"):
            raise ValueError(f"unknown agg strategy: {self.agg!r}")

    def build(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return build_windows(features, labels, window=self.window, stride=self.stride, agg=self.agg)


def build_windows(
    features: np.ndarray,
    labels: np.ndarray | None,
    *,
    window: int,
    stride: int,
    agg: AggStrategy = "last",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(windows, win_labels)``.

    Parameters
    ----------
    features : (N, F) float array.
    labels   : (N,)   int array or None. When None, ``win_labels`` is None too.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    n, f = features.shape
    if n < window:
        raise ValueError(f"not enough rows ({n}) for window size {window}")

    # Vectorised sliding view (NumPy 1.20+).
    sliding = np.lib.stride_tricks.sliding_window_view(features, window_shape=window, axis=0)
    # sliding shape: (n - window + 1, F, W)  -> reorder to (n - window + 1, W, F)
    sliding = np.ascontiguousarray(np.transpose(sliding, (0, 2, 1)))
    sliding = sliding[::stride]
    sliding = sliding.astype(np.float32, copy=False)

    if labels is None:
        return sliding, None

    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(f"labels shape {labels.shape} does not match features {features.shape}")
    win_y_full = np.lib.stride_tricks.sliding_window_view(labels, window_shape=window)
    win_y_full = win_y_full[::stride]

    if agg == "last":
        win_y = win_y_full[:, -1]
    elif agg == "any":
        win_y = (win_y_full.sum(axis=1) > 0).astype(np.int64)
    elif agg == "majority":
        win_y = (win_y_full.sum(axis=1) > (window // 2)).astype(np.int64)
    else:  # pragma: no cover — guarded in __post_init__
        raise AssertionError("unreachable")

    return sliding, win_y.astype(np.int64, copy=False)
