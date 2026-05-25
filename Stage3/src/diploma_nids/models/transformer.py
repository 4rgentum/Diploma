"""Transformer-encoder baseline — Vaswani et al. (2017) self-attention stack."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .base import BaseDeepModel, register


class _SinusoidalPosEnc(nn.Module):
    """Standard sinusoidal positional encoding (Attention Is All You Need §3.5)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(BaseDeepModel):
    name = "transformer"

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        head_hidden: int = 32,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = _SinusoidalPosEnc(d_model, max_len=max(window, 64))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        # mean pooling over time
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


@register("transformer", family="dl")
def _build_transformer(cfg: dict[str, Any]) -> TransformerEncoder:
    p = cfg.get("params", {})
    return TransformerEncoder(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        d_model=int(p.get("d_model", 64)),
        n_heads=int(p.get("n_heads", 4)),
        ff_dim=int(p.get("ff_dim", 128)),
        layers=int(p.get("layers", 2)),
        dropout=float(p.get("dropout", 0.1)),
        head_hidden=int(p.get("head_hidden", 32)),
    )
