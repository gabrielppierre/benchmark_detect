"""Placeholder detector training entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from towervision.utils.io import write_json


def train_detector(
    config: Mapping[str, Any],
    *,
    split_path: Path,
    output_path: Path,
) -> Path:
    """Persist a placeholder detector artifact."""

    artifact = {
        "status": "placeholder",
        "task": "detection",
        "model_name": config.get("name", "detector"),
        "backend": config.get("backend", "placeholder"),
        "split_path": split_path.as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_path, artifact)
    return output_path
