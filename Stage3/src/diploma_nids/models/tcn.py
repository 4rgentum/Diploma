"""Temporal Convolutional Network (Bai, Kolter & Koltun 2018).

A residual block of two causal dilated 1D convolutions; dilation grows
exponentially with depth. Causal padding is achieved by left-padding the input
and trimming the right side after the convolution — same trick used in the
original reference implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import BaseDeepModel, register


class _Chomp1d(nn.Module):
    """Trim the right-padding added for causality."""

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp == 0 else x[:, :, : -self.chomp]


class _TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.act(self.chomp1(self.conv1(x))))
        y = self.dropout(self.act(self.chomp2(self.conv2(y))))
        res = x if self.residual is None else self.residual(x)
        return self.act(y + res)


class TCN(BaseDeepModel):
    name = "tcn"

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        channels: tuple[int, ...] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
        head_hidden: int = 32,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)
        blocks: list[nn.Module] = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            blocks.append(
                _TCNBlock(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = out_ch
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) -> (B, F, W)
        h = self.body(x.transpose(1, 2))
        h = h[:, :, -1]  # last timestep
        return self.head(h).squeeze(-1)


@register("tcn", family="dl")
def _build_tcn(cfg: dict[str, Any]) -> TCN:
    p = cfg.get("params", {})
    return TCN(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        channels=tuple(p.get("channels", (64, 64, 64))),
        kernel_size=int(p.get("kernel_size", 3)),
        dropout=float(p.get("dropout", 0.2)),
        head_hidden=int(p.get("head_hidden", 32)),
    )
