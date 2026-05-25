"""Quick drift-monitor sanity check: compute PSI/KL/MMD between two halves
of a randomly drifted sample. Used as a smoke test, not as the formal E5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.eval import drift_report  # noqa: E402
from diploma_nids.utils import dump_json, ensure_dir, get_logger, set_seed  # noqa: E402

LOG = get_logger("scripts.06_drift_test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--intensity", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "runs" / "drift_smoke.json")
    args = parser.parse_args()

    set_seed(42)
    train_npz = np.load(args.data / "train_windows.npz")
    X = train_npz["windows"]
    rng = np.random.default_rng(0)

    # Reference: 5000 normal windows.
    y = train_npz["labels"]
    ref_idx = np.flatnonzero(y == 0)
    rng.shuffle(ref_idx)
    ref = X[ref_idx[:5000]].reshape(-1, X.shape[-1])
    cur = ref.copy()
    cur += args.intensity * rng.normal(size=cur.shape) * cur.std(axis=0, keepdims=True)

    report = drift_report(ref, cur)
    LOG.info("PSI mean=%.4f  alarm=%s  MMD=%.4f", report["psi"]["mean"], report["drift_alarm"], report["mmd"])
    ensure_dir(args.out.parent)
    dump_json(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
