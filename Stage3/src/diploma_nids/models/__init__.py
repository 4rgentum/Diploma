"""Model registry — single decorator-based namespace for every architecture."""

from .base import (
    BaseDeepModel,
    BaseModel,
    ModelOutput,
    available_models,
    build_model,
    register,
)

# Side-effect imports register every architecture at module-load time.
from . import autoencoder  # noqa: F401
from . import classical  # noqa: F401
from . import cnn1d  # noqa: F401
from . import cnn_lstm  # noqa: F401
from . import mlp  # noqa: F401
from . import rnn_family  # noqa: F401
from . import tcn  # noqa: F401
from . import transformer  # noqa: F401

__all__ = [
    "BaseModel",
    "BaseDeepModel",
    "ModelOutput",
    "available_models",
    "build_model",
    "register",
]
