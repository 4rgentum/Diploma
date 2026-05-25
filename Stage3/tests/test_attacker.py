from __future__ import annotations

from pathlib import Path

import pandas as pd

from diploma_nids.attacker import (
    AttackerRuntime,
    FSMAgent,
    default_template_registry,
    runtime_from_config,
)
from diploma_nids.attacker.agent import FSM_STATES, FSMConfig
from diploma_nids.attacker.drift_injector import DriftInjector, DriftType
from diploma_nids.attacker.templates import build_templates_from_dataframe


ROOT = Path(__file__).resolve().parents[1]


def _trivial_config() -> FSMConfig:
    transitions = {s: {s: 1.0} for s in FSM_STATES}
    return FSMConfig(transitions=transitions, min_durations={}, initial_state="NORMAL")


def test_fsm_emits_n_ticks() -> None:
    reg = default_template_registry()
    agent = FSMAgent(_trivial_config(), reg, seed=0)
    ticks = list(agent.stream(20))
    assert len(ticks) == 20
    assert all(t.fsm_state == "NORMAL" for t in ticks)


def test_fsm_transition_probabilities_validated() -> None:
    import pytest

    bad = FSMConfig(transitions={"NORMAL": {"NORMAL": 0.5}}, initial_state="NORMAL")
    with pytest.raises(ValueError):
        FSMAgent(bad, default_template_registry())


def test_runtime_collects_dataframe(toy_unsw_df: pd.DataFrame) -> None:
    reg = default_template_registry()
    agent = FSMAgent(_trivial_config(), reg, seed=0)
    rt = AttackerRuntime(agent=agent)
    df = rt.collect_dataframe(30)
    assert len(df) == 30
    assert "fsm_state" in df.columns
    assert "label" in df.columns


def test_drift_injector_shifts_numeric(toy_unsw_df: pd.DataFrame) -> None:
    di = DriftInjector(drift_type=DriftType.COVARIATE, intensity=0.5, seed=0)
    sample = {"dur": 1.0, "sbytes": 100.0, "rate": 10.0, "label": 0}
    shifted = di.apply(sample)
    # Some numeric field should have changed.
    assert shifted != sample


def test_empirical_templates_build(toy_unsw_df: pd.DataFrame) -> None:
    reg = build_templates_from_dataframe(toy_unsw_df)
    # At least the dominant categories should be registered.
    names = reg.names()
    assert any(n in ("normal", "dos") for n in names)


def test_runtime_from_config(tmp_path: Path) -> None:
    policy = ROOT / "configs" / "attacker" / "policy.yaml"
    rt = runtime_from_config(policy, seed=42)
    df = rt.collect_dataframe(50)
    assert len(df) == 50
    assert set(df["fsm_state"].unique()).issubset(set(FSM_STATES))
