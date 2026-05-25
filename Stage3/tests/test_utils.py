from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from diploma_nids.utils import dump_json, ensure_dir, load_json, load_yaml, save_yaml, set_seed


def test_set_seed_reproduces_numpy_and_python(tmp_path: Path) -> None:
    set_seed(42)
    a = (random.random(), float(np.random.random()))
    set_seed(42)
    b = (random.random(), float(np.random.random()))
    assert a == b


def test_ensure_dir_creates_parents(tmp_path: Path) -> None:
    p = ensure_dir(tmp_path / "a" / "b" / "c")
    assert p.is_dir()


def test_yaml_and_json_roundtrip(tmp_path: Path) -> None:
    p_yaml = tmp_path / "x.yaml"
    p_json = tmp_path / "x.json"
    payload = {"k": 1, "list": [1, 2.5, "z"], "nested": {"a": True}}
    save_yaml(payload, p_yaml)
    dump_json(payload, p_json)
    assert load_yaml(p_yaml) == payload
    assert load_json(p_json) == payload


def test_dump_json_handles_numpy(tmp_path: Path) -> None:
    p = tmp_path / "n.json"
    dump_json({"a": np.float32(1.5), "b": np.array([1, 2, 3])}, p)
    payload = load_json(p)
    assert payload == {"a": 1.5, "b": [1, 2, 3]}
