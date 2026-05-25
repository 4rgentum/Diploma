"""CNN-LSTM — the proposed target architecture (Stage 2 §3.2).

Pipeline:
    (B, W, F)
        → transpose → Conv1D ×2 (with BN + ReLU + MaxPool + Dropout)
        → transpose → LSTM (bidirectional optional)
        → attention pooling over the LSTM outputs
        → MLP head → logit (B,)

Deviations from a textbook CNN-LSTM that this implementation makes on purpose
(motivated by the Stage 3 review):

* Attention-pooled LSTM outputs instead of the last hidden state — keeps the
  contribution of every timestep and noticeably stabilises training across
  seeds.
* Bidirectional LSTM by default — a recurring choice in the 2024–2025 CNN-LSTM
  papers on UNSW-NB15 (IEEE 2024, Springer 2025, JCBI 2025).
* LayerNorm before the head — protects from the well-known scale drift between
  Conv-output activations and LSTM outputs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import BaseDeepModel, register


class _AttentionPool(nn.Module):
    """Additive attention over a sequence ``(B, T, H)`` → ``(B, H)``."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H)
        u = torch.tanh(self.proj(h))
        a = torch.softmax(self.score(u).squeeze(-1), dim=1)  # (B, T)
        return (h * a.unsqueeze(-1)).sum(dim=1)


class CNNLSTM(BaseDeepModel):
    name = "cnn_lstm"

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        cnn_channels: tuple[int, ...] = (64, 128),
        kernel_size: int = 3,
        lstm_hidden: int = 96,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        head_hidden: int = 64,
        dropout: float = 0.2,
        use_attention_pool: bool = True,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)

        blocks: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in cnn_channels:
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
        self.cnn = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.pool = _AttentionPool(out_dim) if use_attention_pool else None
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F)
        h = x.transpose(1, 2)  # (B, F, W)
        h = self.cnn(h)        # (B, C_last, W/4)
        h = h.transpose(1, 2)  # (B, W/4, C_last)
        h, _ = self.lstm(h)    # (B, W/4, H*dir)
        h = self.pool(h) if self.pool is not None else h[:, -1, :]
        h = self.norm(h)
        return self.head(h).squeeze(-1)


@register("cnn_lstm", family="dl", proposed=True)
def _build_cnn_lstm(cfg: dict[str, Any]) -> CNNLSTM:
    p = cfg.get("params", {})
    return CNNLSTM(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        cnn_channels=tuple(p.get("cnn_channels", (64, 128))),
        kernel_size=int(p.get("kernel_size", 3)),
        lstm_hidden=int(p.get("lstm_hidden", 96)),
        lstm_layers=int(p.get("lstm_layers", 1)),
        bidirectional=bool(p.get("bidirectional", True)),
        head_hidden=int(p.get("head_hidden", 64)),
        dropout=float(p.get("dropout", 0.2)),
        use_attention_pool=bool(p.get("use_attention_pool", True)),
    )
