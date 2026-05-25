"""Stateful streaming scorer.

Accepts one row at a time, buffers them up to the window size, then emits a
score whenever a full window with the proper stride is available. The
``Preprocessor`` is applied incrementally — same artefact as in training.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np
import pandas as pd

from ..data.preprocess import Preprocessor
from ..models.base import BaseModel


class WindowScorer:
    """Sliding-window scorer for online inference."""

    def __init__(
        self,
        *,
        model: BaseModel,
        preprocessor: Preprocessor,
        window: int = 32,
        stride: int = 8,
    ) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        self.model = model
        self.preprocessor = preprocessor
        self.window = int(window)
        self.stride = int(stride)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)
        self._since_last = 0

    def push(self, row: dict | pd.Series) -> float | None:
        """Push a single flow record. Returns a score when a window is ready."""
        if isinstance(row, dict):
            row = pd.Series(row)
        df = pd.DataFrame([row])
        x = self.preprocessor.transform(df)  # (1, F)
        self._buffer.append(x[0])
        self._since_last += 1
        if len(self._buffer) < self.window:
            return None
        if self._since_last < self.stride:
            return None
        self._since_last = 0
        window = np.stack(list(self._buffer), axis=0)[np.newaxis, ...]  # (1, W, F)
        proba = float(self.model.predict_proba(window)[0])
        return proba

    def score_batch(self, df: pd.DataFrame) -> Iterator[tuple[int, float]]:
        """Yield ``(end_row_index, score)`` pairs for an offline DataFrame."""
        x = self.preprocessor.transform(df)
        if x.shape[0] < self.window:
            return
        from ..data.windowing import build_windows

        windows, _ = build_windows(x, None, window=self.window, stride=self.stride)
        probs = self.model.predict_proba(windows)
        for i, p in enumerate(probs):
            end_idx = self.window - 1 + i * self.stride
            yield int(end_idx), float(p)
