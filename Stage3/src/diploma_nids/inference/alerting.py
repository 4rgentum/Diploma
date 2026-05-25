"""Alert formation — five-level severity, time-window deduplication.

The severity ladder (Stage 2 §5.2) buckets ``score / threshold`` into five
human-meaningful classes. Deduplication is a sliding-window aggregator keyed
on ``(src_ip, dst_ip, attack_cat)`` — within ``dedup_seconds`` repeats of the
same triple are folded into the original alert and only the maximum severity
is reported.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


def severity_from_ratio(ratio: float) -> Severity:
    """``ratio = score / threshold`` → severity level."""
    if ratio < 1.0:
        return Severity.INFO  # below threshold should not normally reach AlertFormer
    if ratio < 1.1:
        return Severity.INFO
    if ratio < 1.3:
        return Severity.LOW
    if ratio < 1.6:
        return Severity.MEDIUM
    if ratio < 2.0:
        return Severity.HIGH
    return Severity.CRITICAL


@dataclass
class Alert:
    ts: float
    score: float
    threshold: float
    severity: Severity
    src_ip: str | None = None
    dst_ip: str | None = None
    attack_cat: str | None = None
    dedup_count: int = 1
    model_version: str | None = None
    ground_truth: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


class AlertFormer:
    """Stateful alert builder with sliding-window dedup."""

    def __init__(
        self,
        *,
        threshold: float,
        dedup_seconds: float = 5.0,
        history_size: int = 1024,
        model_version: str | None = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.dedup_seconds = float(dedup_seconds)
        self.history: deque[Alert] = deque(maxlen=int(history_size))
        self.model_version = model_version

    def maybe_emit(
        self,
        *,
        ts: float,
        score: float,
        src_ip: str | None = None,
        dst_ip: str | None = None,
        attack_cat: str | None = None,
        ground_truth: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Return a fresh alert (or update an existing one and return None)."""
        if score < self.threshold:
            return None
        sev = severity_from_ratio(score / self.threshold)

        # Try to merge with the most recent matching alert.
        for prev in reversed(self.history):
            if (ts - prev.ts) > self.dedup_seconds:
                break
            if (prev.src_ip, prev.dst_ip, prev.attack_cat) == (src_ip, dst_ip, attack_cat):
                prev.dedup_count += 1
                if _sev_index(sev) > _sev_index(prev.severity):
                    prev.severity = sev
                    prev.score = max(prev.score, float(score))
                return None

        alert = Alert(
            ts=float(ts),
            score=float(score),
            threshold=self.threshold,
            severity=sev,
            src_ip=src_ip,
            dst_ip=dst_ip,
            attack_cat=attack_cat,
            dedup_count=1,
            model_version=self.model_version,
            ground_truth=ground_truth,
            extras=dict(extras or {}),
        )
        self.history.append(alert)
        return alert

    def recent(self, n: int | None = None) -> list[Alert]:
        n = n if n is not None else len(self.history)
        return list(self.history)[-int(n) :]


def _sev_index(s: Severity) -> int:
    return _SEVERITY_ORDER.index(s)
