"""Stage 4 — aggregate every experiment into the final summary tables.

Reads the raw per-run JSON artefacts in ``Stage3/experiments/runs/`` and
combines them with the saved ``E1_summary.json`` to produce the canonical
result tables under ``Stage4/results/tables/``.

This is the single source of truth for the report — every CSV the report
references is generated from this script. The script is idempotent: running
it twice produces the same files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "Stage3"
STAGE4 = ROOT / "Stage4"
sys.path.insert(0, str(STAGE3 / "src"))

from diploma_nids.utils import dump_json, ensure_dir, get_logger  # noqa: E402

LOG = get_logger("Stage4.build_summary")


def _round_dict(d: dict, places: int = 6) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, places)
        elif isinstance(v, dict):
            out[k] = _round_dict(v, places)
        else:
            out[k] = v
    return out


def main() -> int:
    tables = ensure_dir(STAGE4 / "results" / "tables")
    src_summary = tables / "E1_summary.json"
    if not src_summary.exists():
        LOG.error("E1_summary.json missing; run run_e1_final.py first.")
        return 1

    with src_summary.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)

    df = pd.DataFrame(rows).sort_values("f1_max", ascending=False)
    df.to_csv(tables / "E1_summary.csv", index=False)
    LOG.info("E1 summary written: %d rows", len(df))
    LOG.info(
        "\n%s",
        df[["model", "seeds", "f1_max", "pr_auc", "operating_f1", "operating_fpr"]].to_string(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
