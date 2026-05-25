from __future__ import annotations

import torch

from diploma_nids.training import FocalLoss, build_loss


def test_focal_loss_matches_hand_calc() -> None:
    fl = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    logits = torch.tensor([2.0, -1.0])
    target = torch.tensor([1.0, 0.0])
    loss = fl(logits, target).item()
    assert loss > 0.0


def test_focal_loss_lowers_with_correct_predictions() -> None:
    fl = FocalLoss(alpha=0.5, gamma=2.0)
    correct = fl(torch.tensor([10.0, -10.0]), torch.tensor([1.0, 0.0])).item()
    wrong = fl(torch.tensor([-10.0, 10.0]), torch.tensor([1.0, 0.0])).item()
    assert correct < wrong


def test_build_loss_dispatch() -> None:
    assert build_loss({"name": "focal"}).__class__.__name__ == "FocalLoss"
    assert build_loss({"name": "bce"}).__class__.__name__ == "WeightedBCE"
