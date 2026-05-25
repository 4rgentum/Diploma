"""LSTM / GRU / BiLSTM variants — recurrent baselines."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import BaseDeepModel, register


class _RNNBase(BaseDeepModel):
    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        hidden: int,
        layers: int,
        cell: str,
        bidirectional: bool,
        head_hidden: int,
        dropout: float,
        use_attention_pool: bool,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)
        cell = cell.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        if use_attention_pool:
            self.attn_proj = nn.Linear(out_dim, out_dim)
            self.attn_score = nn.Linear(out_dim, 1)
        else:
            self.attn_proj = self.attn_score = None  # type: ignore[assignment]
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        if self.attn_proj is not None and self.attn_score is not None:
            u = torch.tanh(self.attn_proj(h))
            a = torch.softmax(self.attn_score(u).squeeze(-1), dim=1)
            v = (h * a.unsqueeze(-1)).sum(dim=1)
        else:
            v = h[:, -1, :]
        v = self.norm(v)
        return self.head(v).squeeze(-1)


@register("lstm", family="dl")
def _build_lstm(cfg: dict[str, Any]) -> _RNNBase:
    p = cfg.get("params", {})
    return _RNNBase(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        hidden=int(p.get("hidden", 96)),
        layers=int(p.get("layers", 1)),
        cell="lstm",
        bidirectional=bool(p.get("bidirectional", False)),
        head_hidden=int(p.get("head_hidden", 32)),
        dropout=float(p.get("dropout", 0.2)),
        use_attention_pool=bool(p.get("use_attention_pool", False)),
    )


@register("gru", family="dl")
def _build_gru(cfg: dict[str, Any]) -> _RNNBase:
    p = cfg.get("params", {})
    return _RNNBase(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        hidden=int(p.get("hidden", 96)),
        layers=int(p.get("layers", 1)),
        cell="gru",
        bidirectional=bool(p.get("bidirectional", False)),
        head_hidden=int(p.get("head_hidden", 32)),
        dropout=float(p.get("dropout", 0.2)),
        use_attention_pool=bool(p.get("use_attention_pool", False)),
    )


@register("bilstm", family="dl")
def _build_bilstm(cfg: dict[str, Any]) -> _RNNBase:
    p = cfg.get("params", {})
    return _RNNBase(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        hidden=int(p.get("hidden", 96)),
        layers=int(p.get("layers", 1)),
        cell="lstm",
        bidirectional=True,
        head_hidden=int(p.get("head_hidden", 32)),
        dropout=float(p.get("dropout", 0.2)),
        use_attention_pool=bool(p.get("use_attention_pool", True)),
    )
