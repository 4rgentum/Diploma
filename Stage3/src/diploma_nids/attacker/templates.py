"""Parametrised attack templates.

Each template emits ``FlowRecord``-compatible dicts that match the UNSW-NB15
schema. The Stage 3 review of the previous prototype showed that purely
parametric templates were OOD relative to UNSW (PSI ≈ 0.31 even at drift
intensity 0). To close that gap, templates here support **two operating modes**:

1. **Empirical mode** (preferred) — the template is fitted on the matching
   subset of UNSW-NB15 (``attack_cat == <cat>``) and emits rows by sampling
   each numeric feature from its kernel-density / quantile distribution and
   each categorical feature from its empirical mass function. The result has,
   by construction, near-zero PSI vs the source distribution.

2. **Parametric mode** (fallback) — used when the empirical reference is not
   available. The template still produces semantically meaningful rows (DDoS:
   high rate / SYN flag / small dbytes etc.) but with hand-crafted ranges.

The default registry (``default_template_registry``) is parametric. The
``build_templates_from_dataframe`` helper builds the empirical version from a
UNSW DataFrame and is what the FSM agent uses in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data.schema import (
    UNSW_BINARY,
    UNSW_CATEGORICAL,
    UNSW_FEATURES,
    UNSW_NUMERIC,
)


@dataclass
class AttackTemplate:
    """Single attack template — parametric or empirical."""

    name: str
    attack_cat: str
    fn: Callable[[np.random.Generator], dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.fn(rng)


# --------------------------------------------------------------------------- #
# Empirical mode — fit a per-feature distribution on a labelled subset.        #
# --------------------------------------------------------------------------- #


def _fit_numeric(values: np.ndarray) -> Callable[[np.random.Generator], float]:
    """Build a sampler from the empirical CDF (quantile-based)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return lambda rng: 0.0
    sorted_vals = np.sort(arr)

    def sampler(rng: np.random.Generator) -> float:
        # Inverse-CDF sampling with small Gaussian jitter for smoothness.
        u = float(rng.uniform(0.0, 1.0))
        idx = int(u * (sorted_vals.size - 1))
        base = sorted_vals[idx]
        # Jitter scale ~ IQR/100 to avoid duplicate rows.
        q25, q75 = np.quantile(sorted_vals, [0.25, 0.75])
        jitter = (q75 - q25) / 100.0 if q75 > q25 else 0.0
        return float(base + rng.normal(0.0, jitter) if jitter > 0 else base)

    return sampler


def _fit_categorical(values: pd.Series) -> Callable[[np.random.Generator], str]:
    freq = values.astype(str).str.strip().str.lower().value_counts(normalize=True)
    cats = list(freq.index)
    probs = freq.to_numpy()
    if not cats:
        return lambda rng: "unknown"

    def sampler(rng: np.random.Generator) -> str:
        idx = int(rng.choice(len(cats), p=probs))
        return cats[idx]

    return sampler


def _fit_binary(values: pd.Series) -> Callable[[np.random.Generator], int]:
    p = float((pd.to_numeric(values, errors="coerce").fillna(0) > 0).mean())

    def sampler(rng: np.random.Generator) -> int:
        return int(rng.uniform() < p)

    return sampler


def build_templates_from_dataframe(
    df: pd.DataFrame,
    *,
    category_to_template: dict[str, str] | None = None,
) -> "TemplateRegistry":
    """Build an empirical template per attack_cat present in ``df``.

    ``category_to_template`` lets callers rename attack categories to a custom
    template name (e.g. ``Reconnaissance -> port_scan``). When omitted, the
    template name equals the lowercased ``attack_cat``.
    """
    if "attack_cat" not in df.columns:
        raise ValueError("dataframe must contain 'attack_cat' column")

    registry = TemplateRegistry()
    cat_col = df["attack_cat"].astype(str).str.strip()
    for cat in cat_col.unique():
        sub = df[cat_col == cat]
        if len(sub) < 10:
            # Not enough rows to build a meaningful empirical distribution.
            continue
        # Per-feature samplers, frozen for this category.
        numeric_samplers = {col: _fit_numeric(sub[col].to_numpy()) for col in UNSW_NUMERIC if col in sub.columns}
        cat_samplers = {col: _fit_categorical(sub[col]) for col in UNSW_CATEGORICAL if col in sub.columns}
        bin_samplers = {col: _fit_binary(sub[col]) for col in UNSW_BINARY if col in sub.columns}

        def make_fn(num=numeric_samplers, cats=cat_samplers, bins=bin_samplers, c=cat):
            def fn(rng: np.random.Generator) -> dict[str, Any]:
                row: dict[str, Any] = {}
                for col, s in num.items():
                    row[col] = s(rng)
                for col, s in cats.items():
                    row[col] = s(rng)
                for col, s in bins.items():
                    row[col] = s(rng)
                row["attack_cat"] = c
                row["label"] = 0 if c.lower() == "normal" else 1
                return row

            return fn

        template_name = (category_to_template or {}).get(cat, cat.lower())
        registry.register(AttackTemplate(name=template_name, attack_cat=cat, fn=make_fn()))
    return registry


# --------------------------------------------------------------------------- #
# Parametric fallback templates (kept compact — used only when empirical is   #
# unavailable, e.g. before the dataset has been downloaded).                  #
# --------------------------------------------------------------------------- #


def _empty_row(attack_cat: str = "Normal", label: int = 0) -> dict[str, Any]:
    row = {f: 0.0 for f in UNSW_NUMERIC}
    row.update({f: 0 for f in UNSW_BINARY})
    row.update({f: "unknown" for f in UNSW_CATEGORICAL})
    row["attack_cat"] = attack_cat
    row["label"] = label
    return row


def _parametric_normal(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Normal", 0)
    row["dur"] = float(rng.exponential(2.0))
    row["sbytes"] = float(rng.exponential(800.0))
    row["dbytes"] = float(rng.exponential(1200.0))
    row["spkts"] = int(rng.poisson(8))
    row["dpkts"] = int(rng.poisson(10))
    row["rate"] = float(rng.exponential(50.0))
    row["sttl"] = int(rng.choice([62, 64, 252, 254]))
    row["dttl"] = int(rng.choice([60, 62, 64]))
    row["proto"] = str(rng.choice(["tcp", "udp", "icmp"], p=[0.75, 0.20, 0.05]))
    row["service"] = str(rng.choice(["http", "dns", "smtp", "ssh", "ftp", "-"], p=[0.55, 0.1, 0.05, 0.05, 0.05, 0.20]))
    row["state"] = str(rng.choice(["fin", "con", "int", "req"], p=[0.7, 0.2, 0.05, 0.05]))
    return row


def _parametric_ddos(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("DoS", 1)
    row["dur"] = float(rng.uniform(0.001, 0.05))
    row["sbytes"] = float(rng.uniform(40, 80))   # SYN packets are tiny
    row["dbytes"] = float(rng.uniform(0, 40))
    row["spkts"] = int(rng.poisson(2))
    row["dpkts"] = int(rng.poisson(0.5))
    row["rate"] = float(rng.uniform(2_000, 50_000))
    row["sttl"] = int(rng.choice([254, 252]))
    row["proto"] = "tcp"
    row["service"] = "-"
    row["state"] = "int"
    row["ct_srv_dst"] = int(rng.integers(20, 80))
    return row


def _parametric_scan(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Reconnaissance", 1)
    row["dur"] = float(rng.uniform(0.0, 0.5))
    row["sbytes"] = float(rng.uniform(40, 120))
    row["dbytes"] = float(rng.uniform(0, 60))
    row["spkts"] = int(rng.poisson(2))
    row["dpkts"] = int(rng.poisson(1))
    row["rate"] = float(rng.uniform(50, 1500))
    row["sttl"] = 254
    row["proto"] = str(rng.choice(["tcp", "udp"], p=[0.85, 0.15]))
    row["service"] = "-"
    row["state"] = str(rng.choice(["int", "rst", "req"], p=[0.6, 0.2, 0.2]))
    row["ct_dst_ltm"] = int(rng.integers(10, 60))
    row["ct_src_dport_ltm"] = int(rng.integers(5, 30))
    return row


def _parametric_brute(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Exploits", 1)
    row["dur"] = float(rng.uniform(0.5, 5.0))
    row["sbytes"] = float(rng.uniform(200, 1000))
    row["dbytes"] = float(rng.uniform(50, 500))
    row["spkts"] = int(rng.poisson(15))
    row["dpkts"] = int(rng.poisson(10))
    row["rate"] = float(rng.uniform(20, 200))
    row["proto"] = "tcp"
    row["service"] = str(rng.choice(["ssh", "ftp", "http"], p=[0.6, 0.2, 0.2]))
    row["state"] = "fin"
    row["ct_dst_src_ltm"] = int(rng.integers(10, 80))
    return row


def _parametric_exploit(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Exploits", 1)
    row["dur"] = float(rng.uniform(0.05, 2.0))
    row["sbytes"] = float(rng.uniform(1000, 50_000))
    row["dbytes"] = float(rng.uniform(200, 5_000))
    row["spkts"] = int(rng.poisson(20))
    row["dpkts"] = int(rng.poisson(15))
    row["rate"] = float(rng.uniform(50, 1500))
    row["proto"] = "tcp"
    row["service"] = str(rng.choice(["http", "smtp", "-"], p=[0.5, 0.2, 0.3]))
    row["state"] = str(rng.choice(["fin", "rst"], p=[0.7, 0.3]))
    return row


def _parametric_exfil(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Backdoor", 1)
    row["dur"] = float(rng.uniform(5.0, 120.0))
    row["sbytes"] = float(rng.uniform(50, 500))
    row["dbytes"] = float(rng.uniform(20_000, 1_000_000))  # outbound heavy
    row["spkts"] = int(rng.poisson(40))
    row["dpkts"] = int(rng.poisson(60))
    row["rate"] = float(rng.uniform(10, 200))
    row["proto"] = str(rng.choice(["tcp", "udp"], p=[0.85, 0.15]))
    row["service"] = "http"
    row["state"] = "fin"
    return row


def _parametric_intrec(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Reconnaissance", 1)
    row["dur"] = float(rng.uniform(0.5, 5.0))
    row["sbytes"] = float(rng.uniform(100, 800))
    row["dbytes"] = float(rng.uniform(50, 600))
    row["spkts"] = int(rng.poisson(6))
    row["dpkts"] = int(rng.poisson(4))
    row["rate"] = float(rng.uniform(5, 100))
    row["proto"] = "tcp"
    row["service"] = "-"
    row["state"] = "rst"
    row["ct_srv_src"] = int(rng.integers(5, 40))
    return row


def _parametric_worm(rng: np.random.Generator) -> dict[str, Any]:
    row = _empty_row("Worms", 1)
    row["dur"] = float(rng.uniform(0.01, 0.5))
    row["sbytes"] = float(rng.uniform(400, 1500))
    row["dbytes"] = float(rng.uniform(0, 200))
    row["spkts"] = int(rng.poisson(8))
    row["dpkts"] = int(rng.poisson(1))
    row["rate"] = float(rng.uniform(500, 5000))
    row["proto"] = "tcp"
    row["service"] = "-"
    row["state"] = "int"
    row["ct_dst_ltm"] = int(rng.integers(20, 100))
    return row


class TemplateRegistry:
    """Container for ``AttackTemplate`` objects."""

    def __init__(self) -> None:
        self._templates: dict[str, AttackTemplate] = {}

    def register(self, template: AttackTemplate) -> "TemplateRegistry":
        self._templates[template.name] = template
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def get(self, name: str) -> AttackTemplate:
        if name not in self._templates:
            raise KeyError(f"template {name!r} not registered; known: {sorted(self._templates)}")
        return self._templates[name]

    def names(self) -> list[str]:
        return sorted(self._templates)

    def sample(self, name: str, rng: np.random.Generator) -> dict[str, Any]:
        return self.get(name).sample(rng)


def default_template_registry() -> TemplateRegistry:
    """Parametric registry — used when the dataset is not yet available."""
    reg = TemplateRegistry()
    reg.register(AttackTemplate("normal", "Normal", _parametric_normal))
    reg.register(AttackTemplate("ddos_synflood", "DoS", _parametric_ddos))
    reg.register(AttackTemplate("port_scan", "Reconnaissance", _parametric_scan))
    reg.register(AttackTemplate("brute_force", "Exploits", _parametric_brute))
    reg.register(AttackTemplate("exploit_payload", "Exploits", _parametric_exploit))
    reg.register(AttackTemplate("data_exfil", "Backdoor", _parametric_exfil))
    reg.register(AttackTemplate("internal_recon", "Reconnaissance", _parametric_intrec))
    reg.register(AttackTemplate("worm_lateral", "Worms", _parametric_worm))
    return reg
