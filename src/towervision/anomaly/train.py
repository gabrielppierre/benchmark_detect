"""Placeholder anomaly model training."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from towervision.utils.io import write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def train_anomaly_model(
    config: Mapping[str, Any],
    *,
    crops_dir: Path,
    output_path: Path,
) -> Path:
    """Persist a placeholder anomaly model artifact."""

    crop_files = sorted(
        path.name
        for path in crops_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    artifact = {
        "status": "placeholder",
        "task": "anomaly_detection",
        "model_name": config.get("name", "anomaly"),
        "backend": config.get("backend", "placeholder"),
        "crops_dir": crops_dir.as_posix(),
        "num_training_crops": len(crop_files),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_path, artifact)
    return output_path
