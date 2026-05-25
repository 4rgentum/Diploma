"""Controlled distribution-shift injector — three modes, single scalar
intensity (Stage 2 §6.5).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ..data.schema import UNSW_NUMERIC


class DriftType(str, Enum):
    NONE = "none"
    COVARIATE = "covariate"
    PRIOR = "prior"
    CONCEPT = "concept"


class DriftInjector:
    """Apply a configurable shift to a flow record.

    The injector is **stateful only via the RNG** — given the same seed and
    intensity, the same input yields the same output.
    """

    def __init__(
        self,
        *,
        drift_type: DriftType | str = DriftType.NONE,
        intensity: float = 0.0,
        feature_std: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.drift_type = DriftType(drift_type)
        self.intensity = float(intensity)
        self.feature_std = feature_std or {}
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def apply(self, flow: dict[str, Any]) -> dict[str, Any]:
        if self.drift_type == DriftType.NONE or self.intensity <= 0.0:
            return flow
        if self.drift_type == DriftType.COVARIATE:
            return self._apply_covariate(flow)
        if self.drift_type == DriftType.CONCEPT:
            return self._apply_concept(flow)
        # Prior drift is implemented by the FSM policy (different transition
        # probabilities), not at the flow level.
        return flow

    # --------------------------------------------------------------- #
    def _apply_covariate(self, flow: dict[str, Any]) -> dict[str, Any]:
        out = dict(flow)
        alpha = self.intensity
        for col in UNSW_NUMERIC:
            if col not in out:
                continue
            v = float(out[col])
            sigma = self.feature_std.get(col, max(abs(v), 1.0))
            scale = 1.0 + alpha * float(self.rng.uniform(-0.5, 0.5))
            noise = alpha * sigma * float(self.rng.normal(0.0, 1.0))
            out[col] = v * scale + noise
        return out

    def _apply_concept(self, flow: dict[str, Any]) -> dict[str, Any]:
        """Mask attack signatures — for attack flows only.

        We zero a fraction of the ``ct_*`` counters and randomise ``state`` to
        a typical normal value, which is how an evasive adversary would
        obscure the attack pattern in the metadata.
        """
        if int(flow.get("label", 0)) == 0:
            return flow
        out = dict(flow)
        alpha = self.intensity
        if self.rng.uniform() < alpha:
            for col in UNSW_NUMERIC:
                if col.startswith("ct_") and col in out:
                    out[col] = 0
        if self.rng.uniform() < alpha:
            out["state"] = "fin"
        if self.rng.uniform() < alpha:
            out["service"] = "http"
        return out
