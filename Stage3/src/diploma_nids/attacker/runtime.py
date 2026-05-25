"""Attacker runtime — glues the FSM agent and the drift injector together.

Two ways to use:

* ``collect_dataframe(n_ticks)`` for offline experiments (E4 / E5).
* ``stream_with_callback(n_ticks, cb)`` for the demo and the realtime path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from ..utils.io import load_yaml
from .agent import STATE_TO_TEMPLATE, AgentTick, FSMAgent, FSMConfig
from .drift_injector import DriftInjector, DriftType
from .templates import (
    TemplateRegistry,
    build_templates_from_dataframe,
    default_template_registry,
)


@dataclass
class AttackerRuntime:
    agent: FSMAgent
    injector: DriftInjector | None = None
    seed: int = 42
    flow_columns: list[str] = field(default_factory=list)

    def collect_dataframe(self, n_ticks: int) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for tick in self._iter(n_ticks):
            row = dict(tick.flow)
            row["fsm_state"] = tick.fsm_state
            row["drift_type"] = tick.drift_type
            row["intensity"] = tick.intensity
            row["timestamp"] = tick.timestamp
            records.append(row)
        df = pd.DataFrame.from_records(records)
        if self.flow_columns:
            # Ensure consistent column order — important for downstream Preprocessor.
            extra_cols = ["fsm_state", "drift_type", "intensity", "timestamp", "label", "attack_cat"]
            ordered = [c for c in self.flow_columns if c in df.columns]
            ordered.extend([c for c in extra_cols if c in df.columns and c not in ordered])
            df = df[ordered]
        return df

    def stream_with_callback(
        self,
        n_ticks: int,
        callback: Callable[[AgentTick], None],
    ) -> None:
        for tick in self._iter(n_ticks):
            callback(tick)

    def stream(self, n_ticks: int) -> Iterator[AgentTick]:
        yield from self._iter(n_ticks)

    def _iter(self, n_ticks: int) -> Iterator[AgentTick]:
        for tick in self.agent.stream(n_ticks):
            if self.injector is not None:
                drifted = self.injector.apply(tick.flow)
                tick = AgentTick(
                    timestamp=tick.timestamp,
                    flow=drifted,
                    fsm_state=tick.fsm_state,
                    attack_cat=tick.attack_cat,
                    drift_type=self.injector.drift_type.value if self.injector.intensity > 0 else tick.drift_type,
                    intensity=self.injector.intensity if self.injector.intensity > 0 else tick.intensity,
                )
            yield tick


def runtime_from_config(
    policy_path: str | Path,
    *,
    seed: int = 42,
    empirical_df: pd.DataFrame | None = None,
    drift: dict[str, Any] | None = None,
) -> AttackerRuntime:
    """Build a runtime from a YAML policy and an optional empirical reference.

    ``policy_path`` points to a YAML file with the structure documented in
    ``configs/attacker/policy.yaml``. When ``empirical_df`` is supplied the
    runtime fits empirical templates from it; otherwise it falls back to the
    hand-crafted parametric templates.
    """
    policy = load_yaml(policy_path)
    transitions = policy.get("transitions") or {}
    min_durations = policy.get("min_durations") or {}
    initial = str(policy.get("initial_state", "NORMAL"))
    state_to_tmpl = policy.get("state_to_template") or STATE_TO_TEMPLATE

    if empirical_df is not None:
        cat_map = {
            "Normal": "normal",
            "DoS": "ddos_synflood",
            "Reconnaissance": "port_scan",
            "Exploits": "brute_force",
            "Backdoor": "data_exfil",
            "Worms": "worm_lateral",
            "Generic": "exploit_payload",
            "Fuzzers": "exploit_payload",
            "Shellcode": "exploit_payload",
            "Analysis": "port_scan",
        }
        templates: TemplateRegistry = build_templates_from_dataframe(
            empirical_df, category_to_template=cat_map
        )
    else:
        templates = default_template_registry()

    cfg = FSMConfig(
        transitions=transitions,
        min_durations=min_durations,
        initial_state=initial,
        state_to_template=state_to_tmpl,
    )
    agent = FSMAgent(cfg, templates, seed=seed)
    injector: DriftInjector | None = None
    if drift:
        injector = DriftInjector(
            drift_type=DriftType(drift.get("drift_type", "none")),
            intensity=float(drift.get("intensity", 0.0)),
            feature_std=drift.get("feature_std"),
            seed=seed + 1,
        )
    return AttackerRuntime(agent=agent, injector=injector, seed=seed)
