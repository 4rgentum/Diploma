from __future__ import annotations

import pytest

from diploma_nids.inference import AlertFormer, Severity
from diploma_nids.inference.alerting import severity_from_ratio


def test_severity_ladder() -> None:
    assert severity_from_ratio(1.05) == Severity.INFO
    assert severity_from_ratio(1.2) == Severity.LOW
    assert severity_from_ratio(1.5) == Severity.MEDIUM
    assert severity_from_ratio(1.8) == Severity.HIGH
    assert severity_from_ratio(2.5) == Severity.CRITICAL


def test_alert_skipped_below_threshold() -> None:
    af = AlertFormer(threshold=0.5)
    assert af.maybe_emit(ts=0.0, score=0.4) is None
    assert len(af.history) == 0


def test_dedup_merges_close_alerts() -> None:
    af = AlertFormer(threshold=0.5, dedup_seconds=5.0)
    a1 = af.maybe_emit(ts=0.0, score=0.8, src_ip="1.1.1.1", dst_ip="2.2.2.2", attack_cat="DoS")
    a2 = af.maybe_emit(ts=1.0, score=0.85, src_ip="1.1.1.1", dst_ip="2.2.2.2", attack_cat="DoS")
    assert a1 is not None and a2 is None
    assert af.history[0].dedup_count == 2


def test_dedup_does_not_merge_after_window() -> None:
    af = AlertFormer(threshold=0.5, dedup_seconds=1.0)
    af.maybe_emit(ts=0.0, score=0.8, src_ip="x", dst_ip="y", attack_cat="DoS")
    a2 = af.maybe_emit(ts=2.0, score=0.8, src_ip="x", dst_ip="y", attack_cat="DoS")
    assert a2 is not None  # outside window


def test_history_max_size() -> None:
    af = AlertFormer(threshold=0.5, dedup_seconds=0.0, history_size=3)
    for i in range(10):
        af.maybe_emit(ts=float(i), score=0.9, src_ip=f"{i}.x")
    assert len(af.history) == 3


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AlertFormer(threshold=0.0)
