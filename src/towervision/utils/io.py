"""Lightweight I/O helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_directory(path: Path) -> Path:
    """Remove a directory tree and recreate it empty."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any = _MISSING) -> Any:
    """Read JSON from disk."""

    if not path.exists():
        if default is _MISSING:
            raise FileNotFoundError(path)
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk with stable formatting."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_yaml(path: Path, default: Any = _MISSING) -> Any:
    """Read YAML from disk."""

    if not path.exists():
        if default is _MISSING:
            raise FileNotFoundError(path)
        return default
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: Any) -> None:
    """Write YAML to disk."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_text(path: Path, content: str) -> None:
    """Write plain text to disk."""

    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
