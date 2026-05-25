"""Minimal CLI entrypoint — dispatches into scripts/.

Most real work happens in standalone scripts under ``scripts/`` so they can
be invoked directly with explicit arguments. This module exists primarily so
the ``diploma-nids`` console script (declared in pyproject.toml) has a
single, discoverable entrypoint.
"""

from __future__ import annotations

import sys

from . import __version__


_USAGE = """\
diploma-nids — entrypoint stub.

Commands:
  diploma-nids version       Print package version
  diploma-nids models        List registered model names

For actual experiments, invoke the scripts in scripts/ directly, e.g.
  python scripts/03_train.py --model configs/models/cnn_lstm.yaml --train configs/train/full.yaml
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(_USAGE)
        return 0
    cmd = args[0]
    if cmd == "version":
        sys.stdout.write(f"diploma-nids {__version__}\n")
        return 0
    if cmd == "models":
        from .models import available_models

        names = sorted(available_models().keys())
        for n in names:
            sys.stdout.write(f"{n}\n")
        return 0
    sys.stderr.write(f"unknown command: {cmd!r}\n")
    sys.stderr.write(_USAGE)
    return 2
