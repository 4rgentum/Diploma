"""Reproducibility helpers — fix every random source we depend on."""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

_TORCH_AVAILABLE: Final[bool]
try:
    import torch  # noqa: WPS433

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:  # torch is in our deps, but the helper stays usable without it
    _TORCH_AVAILABLE = False


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Pin every random source we rely on.

    Sets PYTHONHASHSEED, the stdlib `random`, NumPy, and (if available) PyTorch
    CPU/GPU generators. With `deterministic=True`, also forces PyTorch to use
    deterministic algorithms where possible — this is the strongest guarantee
    we can give for reproducibility on CPU.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # `warn_only` so models that legitimately use non-deterministic
            # ops (e.g. some pooling on GPU) do not crash; CPU paths stay
            # deterministic regardless.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (RuntimeError, AttributeError):
                pass
