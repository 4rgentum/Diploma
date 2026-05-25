"""Run the FSM attacker for N ticks and dump the resulting flow records.

Used in offline mode by Experiment E4 and as a smoke test for the attacker
runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diploma_nids.attacker import runtime_from_config  # noqa: E402
from diploma_nids.data import load_unsw_nb15  # noqa: E402
from diploma_nids.utils import ensure_dir, get_logger, load_yaml  # noqa: E402

LOG = get_logger("scripts.05_run_attacker")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "attacker" / "policy.yaml")
    parser.add_argument("--data-cfg", type=Path, default=ROOT / "configs" / "data" / "unsw_nb15.yaml")
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-empirical", action="store_true", default=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "attacker_run.csv")
    args = parser.parse_args()

    df_unsw = None
    if args.use_empirical:
        data_cfg = load_yaml(args.data_cfg)
        try:
            train_df, _ = load_unsw_nb15(
                ROOT / data_cfg["data_dir"],
                train_name=data_cfg["train_csv"],
                test_name=data_cfg["test_csv"],
            )
            df_unsw = train_df
            LOG.info("Empirical mode: fitted templates on %d UNSW rows", len(df_unsw))
        except FileNotFoundError:
            LOG.warning("UNSW data not found — falling back to parametric templates")

    rt = runtime_from_config(args.policy, seed=args.seed, empirical_df=df_unsw)
    df = rt.collect_dataframe(args.ticks)
    LOG.info("Generated %d ticks; columns=%d", len(df), df.shape[1])

    ensure_dir(args.out.parent)
    df.to_csv(args.out, index=False)
    LOG.info("Saved -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
