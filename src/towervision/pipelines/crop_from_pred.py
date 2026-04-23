"""ROI extraction from detector predictions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from towervision.data.load import AnnotationRecord, ImageRecord
from towervision.pipelines.crop_from_gt import crop_bbox


def _safe_token(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def crop_from_predictions(
    images_by_id: Mapping[str, ImageRecord],
    predictions: Sequence[AnnotationRecord],
    *,
    output_dir: Path,
    score_threshold: float = 0.0,
    padding: int = 0,
) -> list[dict[str, Any]]:
    """Generate ROI crops from detector predictions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_image: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for prediction in predictions:
        if prediction.score is not None and prediction.score < score_threshold:
            continue
        predictions_by_image[prediction.image_id].append(prediction)

    manifest: list[dict[str, Any]] = []
    for image_id, image_predictions in predictions_by_image.items():
        image_record = images_by_id.get(image_id)
        if image_record is None or not image_record.path.exists():
            continue

        with Image.open(image_record.path) as image:
            for prediction in image_predictions:
                crop = crop_bbox(image, prediction.bbox, padding=padding)
                crop_name = f"{_safe_token(image_id)}__{_safe_token(prediction.id)}__pred.png"
                crop_path = output_dir / crop_name
                crop.save(crop_path)
                manifest.append(
                    {
                        "crop_path": crop_path.as_posix(),
                        "image_id": image_id,
                        "prediction_id": prediction.id,
                        "label": prediction.label,
                        "score": prediction.score,
                        "source": "pred",
                    }
                )

    return manifest
