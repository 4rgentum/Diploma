"""Loss functions for the deep models.

* ``FocalLoss`` — Lin et al. 2017, binary-classification form.
* ``WeightedBCE`` — sample-reweighted BCE with logits.
* ``build_loss`` — dispatch helper used by the trainer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss with optional class balancing.

    ``L = -alpha * (1 - p_t)^gamma * log(p_t)``

    The implementation uses ``binary_cross_entropy_with_logits`` for numerical
    stability and operates directly on logits.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCE(nn.Module):
    """BCE with logits + optional per-class weights."""

    def __init__(self, pos_weight: float | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kw: dict[str, Any] = {"reduction": self.reduction}
        if self.pos_weight is not None:
            kw["pos_weight"] = torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target.float(), **kw)


def build_loss(cfg: dict[str, Any] | None) -> nn.Module:
    cfg = cfg or {}
    name = cfg.get("name", "focal").lower()
    if name in ("focal", "focal_loss"):
        return FocalLoss(
            alpha=float(cfg.get("alpha", 0.25)),
            gamma=float(cfg.get("gamma", 2.0)),
        )
    if name in ("bce", "binary_cross_entropy"):
        return WeightedBCE(pos_weight=cfg.get("pos_weight"))
    raise ValueError(f"unknown loss {name!r}")
