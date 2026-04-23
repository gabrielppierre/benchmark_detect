"""Load and normalize dataset manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from towervision.utils.io import read_json, write_json

BBox = tuple[float, float, float, float]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class ImageRecord:
    """Image metadata used by the pipeline."""

    id: str
    path: Path
    width: int | None = None
    height: int | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ImageRecord":
        return cls(
            id=str(raw["id"]),
            path=Path(raw["path"]),
            width=raw.get("width"),
            height=raw.get("height"),
            split=raw.get("split"),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path.as_posix(),
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AnnotationRecord:
    """Normalized annotation or prediction."""

    id: str
    image_id: str
    bbox: BBox
    label: str = "isolator"
    score: float | None = None
    source: str = "gt"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        default_label: str = "isolator",
        default_source: str = "gt",
    ) -> "AnnotationRecord":
        x, y, width, height = (float(value) for value in raw["bbox"])
        return cls(
            id=str(raw["id"]),
            image_id=str(raw["image_id"]),
            bbox=(x, y, width, height),
            label=str(raw.get("label", default_label)),
            score=raw.get("score"),
            source=str(raw.get("source", default_source)),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "bbox": list(self.bbox),
            "label": self.label,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


def discover_images(images_dir: Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[ImageRecord]:
    """Discover images under a root directory."""

    if not images_dir.exists():
        return []

    allowed_extensions = {extension.lower() for extension in extensions}
    records: list[ImageRecord] = []
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        with Image.open(path) as image:
            width, height = image.size
        image_id = path.relative_to(images_dir).with_suffix("").as_posix()
        records.append(ImageRecord(id=image_id, path=path, width=width, height=height))
    return records


def load_images_manifest(path: Path) -> list[ImageRecord]:
    """Load image records from a JSON manifest."""

    payload = read_json(path, default=[])
    return [ImageRecord.from_dict(item) for item in payload]


def save_images_manifest(path: Path, images: Iterable[ImageRecord]) -> None:
    """Persist image records to JSON."""

    write_json(path, [image.to_dict() for image in images])


def _normalized_image_id(file_name: str) -> str:
    """Create a stable image identifier from a relative file name."""

    return Path(file_name).with_suffix("").as_posix()


def load_annotations(
    path: Path,
    *,
    default_label: str = "isolator",
    default_source: str = "gt",
    allow_missing: bool = False,
) -> list[AnnotationRecord]:
    """Load annotations or predictions from JSON."""

    if allow_missing and not path.exists():
        return []

    payload = read_json(path, default=[])
    if isinstance(payload, dict):
        raw_annotations = payload.get("annotations", [])
    else:
        raw_annotations = payload
    return [
        AnnotationRecord.from_dict(
            item,
            default_label=default_label,
            default_source=default_source,
        )
        for item in raw_annotations
    ]


def load_coco_dataset(
    images_dir: Path,
    annotations_path: Path,
    *,
    default_source: str = "gt",
) -> tuple[list[ImageRecord], list[AnnotationRecord]]:
    """Load a COCO-style dataset and normalize it to local records."""

    payload = read_json(annotations_path, default={})
    if not isinstance(payload, Mapping):
        raise ValueError("COCO annotations must be a JSON object")

    raw_images = payload.get("images", [])
    raw_annotations = payload.get("annotations", [])
    raw_categories = payload.get("categories", [])

    category_by_id = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in raw_categories
    }
    image_id_map: dict[int, str] = {}
    images: list[ImageRecord] = []

    for raw_image in raw_images:
        file_name = str(raw_image["file_name"])
        normalized_id = _normalized_image_id(file_name)
        raw_image_id = int(raw_image["id"])
        image_id_map[raw_image_id] = normalized_id
        metadata = {
            "source_image_id": raw_image_id,
            "file_name": file_name,
            "license": raw_image.get("license"),
            "date_captured": raw_image.get("date_captured"),
        }
        images.append(
            ImageRecord(
                id=normalized_id,
                path=images_dir / file_name,
                width=raw_image.get("width"),
                height=raw_image.get("height"),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    annotations: list[AnnotationRecord] = []
    for raw_annotation in raw_annotations:
        raw_image_id = int(raw_annotation["image_id"])
        image_id = image_id_map.get(raw_image_id, str(raw_image_id))
        metadata = {
            "category_id": raw_annotation.get("category_id"),
            "area": raw_annotation.get("area"),
            "iscrowd": raw_annotation.get("iscrowd"),
            "attributes": raw_annotation.get("attributes"),
            "segmentation": raw_annotation.get("segmentation"),
        }
        annotations.append(
            AnnotationRecord(
                id=str(raw_annotation["id"]),
                image_id=image_id,
                bbox=tuple(float(value) for value in raw_annotation["bbox"]),
                label=category_by_id.get(int(raw_annotation["category_id"]), "unknown"),
                score=raw_annotation.get("score"),
                source=default_source,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    return images, annotations


def save_annotations(path: Path, annotations: Iterable[AnnotationRecord]) -> None:
    """Persist annotations or predictions to JSON."""

    write_json(path, [annotation.to_dict() for annotation in annotations])


def index_images_by_id(images: Iterable[ImageRecord]) -> dict[str, ImageRecord]:
    """Index image records by identifier."""

    return {image.id: image for image in images}
