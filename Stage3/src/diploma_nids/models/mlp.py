"""MLP baseline — flattens the window and applies a small feed-forward stack."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import BaseDeepModel, register


class MLP(BaseDeepModel):
    name = "mlp"
    accepts_windows = True

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)
        in_dim = input_dim * window
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.flatten(1)
        return self.net(x).squeeze(-1)


@register("mlp", family="dl")
def _build_mlp(cfg: dict[str, Any]) -> MLP:
    p = cfg.get("params", {})
    return MLP(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        hidden=tuple(p.get("hidden", (256, 128))),
        dropout=float(p.get("dropout", 0.2)),
    )
