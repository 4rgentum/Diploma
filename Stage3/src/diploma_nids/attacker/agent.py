"""Finite-state attacker agent.

State alphabet (Stage 2 §6.2):
    NORMAL, ATTACK_DDOS, ATTACK_SCAN, ATTACK_BRUTE, ATTACK_EXPLOIT,
    ATTACK_EXFIL, ATTACK_INTREC, ATTACK_WORM, DRIFT_COV, DRIFT_PRIOR,
    DRIFT_CONCEPT.

The agent stays in each state for at least ``min_duration`` ticks before any
transition is allowed. Transitions are sampled from a stochastic policy
(``state → distribution over next states``) loaded from YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np

from .templates import AttackTemplate, TemplateRegistry

FSM_STATES: tuple[str, ...] = (
    "NORMAL",
    "ATTACK_DDOS",
    "ATTACK_SCAN",
    "ATTACK_BRUTE",
    "ATTACK_EXPLOIT",
    "ATTACK_EXFIL",
    "ATTACK_INTREC",
    "ATTACK_WORM",
    "DRIFT_COV",
    "DRIFT_PRIOR",
    "DRIFT_CONCEPT",
)

# Default mapping of state -> template name. Drift states reuse a normal-ish
# template; the actual drift effect is applied by the DriftInjector wrapping
# the runtime.
STATE_TO_TEMPLATE: dict[str, str] = {
    "NORMAL": "normal",
    "ATTACK_DDOS": "ddos_synflood",
    "ATTACK_SCAN": "port_scan",
    "ATTACK_BRUTE": "brute_force",
    "ATTACK_EXPLOIT": "exploit_payload",
    "ATTACK_EXFIL": "data_exfil",
    "ATTACK_INTREC": "internal_recon",
    "ATTACK_WORM": "worm_lateral",
    "DRIFT_COV": "normal",
    "DRIFT_PRIOR": "normal",
    "DRIFT_CONCEPT": "normal",
}


@dataclass
class AgentTick:
    """One agent emission — a single row of synthetic traffic with metadata."""

    timestamp: float
    flow: dict[str, Any]
    fsm_state: str
    attack_cat: str | None
    drift_type: str | None
    intensity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FSMConfig:
    transitions: dict[str, dict[str, float]] = field(default_factory=dict)
    min_durations: dict[str, int] = field(default_factory=dict)
    initial_state: str = "NORMAL"
    state_to_template: dict[str, str] = field(default_factory=lambda: dict(STATE_TO_TEMPLATE))


class FSMAgent:
    """Stochastic FSM with controlled minimum dwell times."""

    def __init__(
        self,
        config: FSMConfig,
        templates: TemplateRegistry,
        *,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.templates = templates
        self.rng = np.random.default_rng(seed)
        self._validate_transitions()

        self.state = config.initial_state
        if self.state not in FSM_STATES:
            raise ValueError(f"initial_state {self.state!r} not in FSM alphabet")
        self._ticks_in_state = 0
        self._tick_counter = 0

    # --------------------------------------------------------------- #
    # public                                                           #
    # --------------------------------------------------------------- #
    def reset(self, *, state: str | None = None, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = state if state is not None else self.config.initial_state
        self._ticks_in_state = 0
        self._tick_counter = 0

    def tick(self) -> AgentTick:
        self._maybe_transition()
        tmpl_name = self.config.state_to_template.get(self.state, "normal")
        template = self._get_template(tmpl_name)
        flow = template.sample(self.rng)
        tick = AgentTick(
            timestamp=float(self._tick_counter),
            flow=flow,
            fsm_state=self.state,
            attack_cat=None if self.state.startswith("DRIFT_") or self.state == "NORMAL" else template.attack_cat,
            drift_type=self.state[len("DRIFT_") :].lower() if self.state.startswith("DRIFT_") else None,
            intensity=0.0,
        )
        self._tick_counter += 1
        self._ticks_in_state += 1
        return tick

    def stream(self, n_ticks: int) -> Iterator[AgentTick]:
        for _ in range(int(n_ticks)):
            yield self.tick()

    # --------------------------------------------------------------- #
    # internals                                                        #
    # --------------------------------------------------------------- #
    def _maybe_transition(self) -> None:
        min_dwell = self.config.min_durations.get(self.state, 1)
        if self._ticks_in_state < min_dwell:
            return
        dist = self.config.transitions.get(self.state)
        if not dist:
            return
        next_states = list(dist.keys())
        probs = np.array([dist[s] for s in next_states], dtype=np.float64)
        if probs.sum() <= 0:
            return
        probs = probs / probs.sum()
        nxt = str(self.rng.choice(next_states, p=probs))
        if nxt != self.state:
            self.state = nxt
            self._ticks_in_state = 0

    def _validate_transitions(self) -> None:
        for src, dist in self.config.transitions.items():
            if src not in FSM_STATES:
                raise ValueError(f"unknown source state {src!r} in transitions")
            for dst in dist:
                if dst not in FSM_STATES:
                    raise ValueError(f"unknown destination state {dst!r} from {src!r}")
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"transition probabilities from {src!r} sum to {total:.4f}, expected 1.0")

    def _get_template(self, name: str) -> AttackTemplate:
        if name in self.templates:
            return self.templates.get(name)
        # Fall back to the normal template if a state-specific one is absent
        # (happens when empirical templates are missing for rare categories).
        if "normal" in self.templates:
            return self.templates.get("normal")
        raise KeyError(f"template {name!r} not in registry and no 'normal' fallback")
