"""Placeholder detector inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from towervision.data.load import AnnotationRecord, ImageRecord, save_annotations


def make_placeholder_prediction(
    image: ImageRecord,
    *,
    score: float,
    label: str = "isolator",
) -> AnnotationRecord:
    """Create a deterministic placeholder prediction for one image."""

    width = max(image.width or 1, 1)
    height = max(image.height or 1, 1)
    x = width // 10
    y = height // 10
    bbox_width = max(1, width - (2 * x))
    bbox_height = max(1, height - (2 * y))
    return AnnotationRecord(
        id=f"pred-{image.id}",
        image_id=image.id,
        bbox=(x, y, bbox_width, bbox_height),
        label=label,
        score=score,
        source="pred",
    )


def infer_detector(
    images: Sequence[ImageRecord],
    *,
    model_artifact: Mapping[str, Any] | None = None,
    confidence_threshold: float = 0.25,
) -> list[AnnotationRecord]:
    """Generate deterministic placeholder predictions."""

    _ = model_artifact
    return [
        make_placeholder_prediction(image, score=confidence_threshold + 0.5)
        for image in images
    ]


def save_predictions(path, predictions: Sequence[AnnotationRecord]) -> None:
    """Persist predictions using the shared annotation schema."""

    save_annotations(path, predictions)
