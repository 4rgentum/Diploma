"""Project-wide logger factory. Single console handler, no duplicates."""

from __future__ import annotations

import logging
import sys
from typing import Final

_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_INITIALISED: dict[str, logging.Logger] = {}


def get_logger(name: str = "diploma_nids", level: int | str = logging.INFO) -> logging.Logger:
    if name in _INITIALISED:
        return _INITIALISED[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)

    _INITIALISED[name] = logger
    return logger
