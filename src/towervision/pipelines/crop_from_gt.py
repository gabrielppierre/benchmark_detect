"""ROI extraction from ground-truth annotations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from towervision.data.load import AnnotationRecord, ImageRecord


def _safe_token(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def crop_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    padding: int = 0,
) -> Image.Image:
    """Crop a bounding box from an image with optional padding."""

    x, y, width, height = bbox
    x0 = max(0, math.floor(x - padding))
    y0 = max(0, math.floor(y - padding))
    x1 = min(image.width, math.ceil(x + width + padding))
    y1 = min(image.height, math.ceil(y + height + padding))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid crop after clamping bbox")
    return image.crop((x0, y0, x1, y1))


def crop_from_ground_truth(
    images_by_id: Mapping[str, ImageRecord],
    annotations: Sequence[AnnotationRecord],
    *,
    output_dir: Path,
    padding: int = 0,
) -> list[dict[str, Any]]:
    """Generate ROI crops from ground-truth boxes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation.image_id].append(annotation)

    manifest: list[dict[str, Any]] = []
    for image_id, image_annotations in annotations_by_image.items():
        image_record = images_by_id.get(image_id)
        if image_record is None or not image_record.path.exists():
            continue

        with Image.open(image_record.path) as image:
            for annotation in image_annotations:
                crop = crop_bbox(image, annotation.bbox, padding=padding)
                crop_name = (
                    f"{_safe_token(image_id)}__{_safe_token(annotation.id)}__{annotation.label}.png"
                )
                crop_path = output_dir / crop_name
                crop.save(crop_path)
                manifest.append(
                    {
                        "crop_path": crop_path.as_posix(),
                        "image_id": image_id,
                        "annotation_id": annotation.id,
                        "label": annotation.label,
                        "source": "gt",
                    }
                )

    return manifest
