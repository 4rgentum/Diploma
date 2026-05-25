"""Autoencoder & Variational Autoencoder — semi-supervised anomaly detectors.

Trained on the normal class only (or with class weighting). At inference,
reconstruction error becomes the anomaly score. We expose it as a
probability via min-max + sigmoid transform on the validation reference
window so the same ``predict_proba`` contract holds.

Implementation notes:

* Inputs ``(B, W, F)`` are flattened to ``(B, W * F)``. AE/VAE intentionally
  ignore the temporal structure — they are baselines that test whether the
  detection works on bag-of-flows representation alone.
* The wrapper also caches a calibration statistic ``ref_score_max`` so the
  output is bounded in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .base import BaseDeepModel, register


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], latent: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent: int, hidden: tuple[int, ...], out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent
        for h in reversed(hidden):
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(BaseDeepModel):
    name = "autoencoder"
    is_supervised = False

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        hidden: tuple[int, ...] = (256, 128),
        latent: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window)
        flat = input_dim * window
        self.flat = flat
        self.encoder = _Encoder(flat, hidden, latent, dropout)
        self.decoder = _Decoder(latent, hidden, flat, dropout)
        # Calibration constant filled in after training (see trainer).
        self.register_buffer("ref_score_max", torch.tensor(1.0))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.flatten(1)
        z = self.encoder(x)
        return self.decoder(z), x  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, target = self.reconstruct(x)
        err = ((recon - target) ** 2).mean(dim=1)
        # Map MSE -> logit so sigmoid(logit) ≈ normalised anomaly probability.
        max_score = self.ref_score_max.clamp_min(1e-6)
        prob = (err / max_score).clamp(0.0, 0.999)
        logit = torch.log(prob / (1.0 - prob).clamp_min(1e-6))
        return logit


class VAE(Autoencoder):
    name = "vae"
    is_supervised = False

    def __init__(
        self,
        *,
        input_dim: int,
        window: int,
        hidden: tuple[int, ...] = (256, 128),
        latent: int = 32,
        dropout: float = 0.0,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(input_dim=input_dim, window=window, hidden=hidden, latent=latent, dropout=dropout)
        self.kl_weight = float(kl_weight)
        self.mu_head = nn.Linear(latent, latent)
        self.logvar_head = nn.Linear(latent, latent)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.flatten(1)
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        recon = self.decoder(z)
        # Stash latent stats on the module so the trainer can read them.
        self._last_mu = mu
        self._last_logvar = logvar
        return recon, x  # type: ignore[return-value]


@register("autoencoder", family="dl", is_supervised=False)
def _build_ae(cfg: dict[str, Any]) -> Autoencoder:
    p = cfg.get("params", {})
    return Autoencoder(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        hidden=tuple(p.get("hidden", (256, 128))),
        latent=int(p.get("latent", 32)),
        dropout=float(p.get("dropout", 0.0)),
    )


@register("vae", family="dl", is_supervised=False)
def _build_vae(cfg: dict[str, Any]) -> VAE:
    p = cfg.get("params", {})
    return VAE(
        input_dim=int(cfg["input_dim"]),
        window=int(cfg.get("window", 32)),
        hidden=tuple(p.get("hidden", (256, 128))),
        latent=int(p.get("latent", 32)),
        dropout=float(p.get("dropout", 0.0)),
        kl_weight=float(p.get("kl_weight", 1.0)),
    )
