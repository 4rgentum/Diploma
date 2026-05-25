"""Help locating the UNSW-NB15 CSVs.

The dataset is not auto-downloaded — the original zip lives in the repo root
(``../unsw.zip``) and the project assumes the user puts the extracted CSVs in
``data/unsw_nb15/``. This script is a small convenience tool that unpacks the
zip if it is present and prints the expected paths.

The current upstream download URLs change over time; the canonical reference
remains https://research.unsw.edu.au/projects/unsw-nb15-dataset.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "data" / "unsw_nb15"
DEFAULT_ZIP = ROOT.parent / "unsw.zip"  # this is where the repo currently keeps it


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpack UNSW-NB15 zip if present.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to unsw.zip archive")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Where to extract the CSVs")
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)

    if (args.target / "UNSW_NB15_training-set.csv").exists():
        print(f"[ok] UNSW-NB15 CSVs already present in {args.target}")
        return 0

    if not args.zip.exists():
        print(
            f"[error] Archive {args.zip} not found. Place the CSVs manually in {args.target}\n"
            "        Reference: https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            file=sys.stderr,
        )
        return 1

    print(f"[info] Extracting {args.zip} → {args.target}")
    with zipfile.ZipFile(args.zip) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            target = args.target / Path(member).name
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    print(f"[ok] Extracted to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
