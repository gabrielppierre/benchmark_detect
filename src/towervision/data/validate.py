"""Validation helpers for dataset annotations."""

from __future__ import annotations

from collections.abc import Sequence

from towervision.data.load import AnnotationRecord, ImageRecord


def validate_annotations(
    images: Sequence[ImageRecord],
    annotations: Sequence[AnnotationRecord],
) -> list[str]:
    """Return validation errors for a set of annotations."""

    image_index = {image.id: image for image in images}
    errors: list[str] = []
    for annotation in annotations:
        image = image_index.get(annotation.image_id)
        if image is None:
            errors.append(f"{annotation.id}: unknown image_id '{annotation.image_id}'")
            continue

        x, y, width, height = annotation.bbox
        if width <= 0 or height <= 0:
            errors.append(f"{annotation.id}: bbox must have positive width and height")
        if x < 0 or y < 0:
            errors.append(f"{annotation.id}: bbox cannot start with negative coordinates")
        if image.width is not None and x + width > image.width:
            errors.append(f"{annotation.id}: bbox exceeds image width")
        if image.height is not None and y + height > image.height:
            errors.append(f"{annotation.id}: bbox exceeds image height")
    return errors


def build_validation_report(
    images: Sequence[ImageRecord],
    annotations: Sequence[AnnotationRecord],
) -> dict[str, object]:
    """Build a compact validation summary."""

    errors = validate_annotations(images, annotations)
    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_errors": len(errors),
        "is_valid": not errors,
        "errors": errors,
    }
