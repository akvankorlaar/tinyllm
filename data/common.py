"""Shared helpers for the data pipeline: config loading and JSONL I/O."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config. Resolve relative paths inside it with resolve()."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a (possibly relative) path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict], append: bool = False) -> int:
    """Write rows as JSONL (one object per line). Returns the count written."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "a" if append else "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    """Yield objects from a JSONL file (plain or .gz)."""
    path = resolve(path)
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
