"""Base classes and registry for every model.

Two layers of abstraction:

* ``BaseModel`` — protocol used by both deep and classical models. Predicts
  ``ModelOutput`` (logits + probabilities + extras) for a 2- or 3-D input.
* ``BaseDeepModel`` — ``nn.Module`` subclass that wires together the forward
  pass for windowed sequence inputs and exposes the same ``predict_proba``
  contract as classical models.

The registry is a tiny decorator-based dispatcher. Every model file calls
``@register("name")`` and provides ``build(cfg)`` that returns an instance.
``build_model(cfg)`` is the public entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH_OK = True
except ModuleNotFoundError:  # pragma: no cover — torch is in requirements
    _TORCH_OK = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class ModelOutput:
    """Standardised inference output."""

    logits: np.ndarray  # (B,)
    probs: np.ndarray  # (B,) in [0, 1]
    extras: dict[str, Any]


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

_BUILDERS: dict[str, Callable[[dict[str, Any]], "BaseModel"]] = {}
_META: dict[str, dict[str, Any]] = {}


def register(name: str, **meta: Any) -> Callable[[Callable[[dict[str, Any]], "BaseModel"]], Callable[[dict[str, Any]], "BaseModel"]]:
    """Decorator: register a builder function under ``name``."""

    def deco(fn: Callable[[dict[str, Any]], "BaseModel"]) -> Callable[[dict[str, Any]], "BaseModel"]:
        if name in _BUILDERS:
            raise ValueError(f"model name {name!r} already registered")
        _BUILDERS[name] = fn
        _META[name] = dict(meta)
        return fn

    return deco


def available_models() -> dict[str, dict[str, Any]]:
    return dict(_META)


def build_model(cfg: dict[str, Any]) -> "BaseModel":
    """Instantiate a model from its YAML config dict.

    ``cfg`` must contain a top-level ``name`` key matching a registered builder.
    """
    if "name" not in cfg:
        raise KeyError("model config must contain a 'name' key")
    name = cfg["name"]
    if name not in _BUILDERS:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name](cfg)


# --------------------------------------------------------------------------- #
# Abstract contracts                                                           #
# --------------------------------------------------------------------------- #


class BaseModel:
    """Minimum protocol every model fulfils."""

    name: ClassVar[str] = "base"
    accepts_windows: ClassVar[bool] = True
    is_supervised: ClassVar[bool] = True
    needs_torch: ClassVar[bool] = False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def output(self, X: np.ndarray) -> ModelOutput:
        p = self.predict_proba(X)
        # Convert to a stable logit. probabilities exactly at 0 or 1 become
        # +/- 30 — keeps downstream calibration stable without inf values.
        eps = 1e-6
        clip = np.clip(p, eps, 1.0 - eps)
        logits = np.log(clip / (1.0 - clip)).astype(np.float32)
        return ModelOutput(logits=logits.astype(np.float32), probs=p.astype(np.float32), extras={})


if _TORCH_OK:

    class BaseDeepModel(nn.Module, BaseModel):  # type: ignore[misc]
        """PyTorch base — windowed input ``(B, W, F)`` → logit ``(B,)``."""

        needs_torch: ClassVar[bool] = True

        def __init__(self, *, input_dim: int, window: int) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.window = int(window)

        # ``forward`` returns a (B,) logit tensor by convention.
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: D401
            raise NotImplementedError

        @torch.no_grad()  # type: ignore[misc]
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.eval()
            x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            logits = self.forward(x)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        @torch.no_grad()  # type: ignore[misc]
        def predict_logits(self, X: np.ndarray) -> np.ndarray:
            self.eval()
            x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            return self.forward(x).cpu().numpy().astype(np.float32)

else:  # pragma: no cover — only triggered if torch is missing

    class BaseDeepModel(BaseModel):  # type: ignore[no-redef]
        needs_torch: ClassVar[bool] = True

        def __init__(self, *, input_dim: int, window: int) -> None:
            raise ImportError("PyTorch is required for deep models")
