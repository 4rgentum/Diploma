from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_ROOT = ROOT / "configs"


@pytest.mark.parametrize("p", sorted(CFG_ROOT.rglob("*.yaml")))
def test_yaml_loads(p: Path) -> None:
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data is not None, f"{p} parsed to None"


def test_attacker_policy_transitions_sum_to_one() -> None:
    p = CFG_ROOT / "attacker" / "policy.yaml"
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for state, dist in data["transitions"].items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-4, f"transitions from {state} sum to {total}"


def test_all_models_have_configs() -> None:
    expected = {
        "mlp", "cnn1d", "cnn_lstm", "lstm", "gru", "bilstm", "tcn", "transformer",
        "autoencoder", "vae",
    }
    available = {p.stem for p in (CFG_ROOT / "models").glob("*.yaml") if p.stem != "classical"}
    missing = expected - available
    assert not missing, f"missing model configs: {missing}"
