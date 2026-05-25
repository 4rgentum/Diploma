"""diploma_nids — NIDS methodology: deep-learning detector + FSM attack agent + drift monitor."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diploma-nids")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
