"""1D-CNN baseline — local pattern extractor over the window axis."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import BaseDeepModel, register


class CNN1D(BaseDeepModel):
    name = "cnn1d"

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        channels: tuple[int, ...] = (64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)
        blocks: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in channels:
            blocks.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) -> (B, F, W) for Conv1d
        x = x.transpose(1, 2)
        x = self.body(x)
        return self.head(x).squeeze(-1)


@register("cnn1d", family="dl")
def _build_cnn1d(cfg: dict[str, Any]) -> CNN1D:
    p = cfg.get("params", {})
    return CNN1D(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        channels=tuple(p.get("channels", (64, 128))),
        kernel_size=int(p.get("kernel_size", 3)),
        dropout=float(p.get("dropout", 0.2)),
    )
