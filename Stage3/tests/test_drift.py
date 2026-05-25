from __future__ import annotations

import numpy as np

from diploma_nids.eval import drift_report, kl_divergence, mmd_rbf, psi


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 0.01


def test_psi_grows_with_shift() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0, scale=1, size=5000)
    b = rng.normal(loc=2, scale=1, size=5000)
    assert psi(a, b) > 0.5


def test_kl_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    assert kl_divergence(x, x) < 0.05


def test_mmd_zero_for_identical() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 4))
    val = mmd_rbf(x, x.copy())
    assert abs(val) < 0.01


def test_drift_report_alarms_on_shift() -> None:
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(2000, 5))
    cur = ref + 1.5
    rep = drift_report(ref, cur)
    assert rep["psi"]["mean"] > 0.25
    assert rep["drift_alarm"]


def test_drift_report_silent_on_identical() -> None:
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(2000, 5))
    rep = drift_report(ref, ref.copy())
    assert rep["psi"]["mean"] < 0.05
    assert not rep["drift_alarm"]
