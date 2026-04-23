"""Materialize the frozen split into reusable benchmark dataset views."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from towervision.data.load import AnnotationRecord, ImageRecord, load_annotations, load_images_manifest
from towervision.detectors.benchmark_types import BenchmarkDatasetArtifacts, CLASS_NAMES
from towervision.utils.io import clean_directory, read_json, write_json, write_yaml

COCO_SPLIT_NAMES = {
    "train": "train2017",
    "val": "val2017",
    "test": "test2017",
}


def _safe_symlink(source: Path, target: Path) -> None:
    """Create or replace a symlink to the source image."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def _group_annotations_by_image(
    annotations: Sequence[AnnotationRecord],
) -> dict[str, list[AnnotationRecord]]:
    """Group annotations by image identifier."""

    annotations_by_image: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation.image_id].append(annotation)
    return annotations_by_image


def _build_category_maps(class_names: Sequence[str]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Build forward and reverse COCO category maps."""

    category_by_label = {label: index + 1 for index, label in enumerate(class_names)}
    categories = [{"id": category_by_label[label], "name": label} for label in class_names]
    return category_by_label, categories


def _annotation_to_coco(
    annotation: AnnotationRecord,
    *,
    image_numeric_id: int,
    annotation_numeric_id: int,
    category_by_label: Mapping[str, int],
) -> dict[str, Any]:
    """Convert a normalized annotation into a COCO annotation record."""

    x, y, width, height = annotation.bbox
    area = float(annotation.metadata.get("area", width * height))
    return {
        "id": annotation_numeric_id,
        "image_id": image_numeric_id,
        "category_id": category_by_label[annotation.label],
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": area,
        "iscrowd": int(annotation.metadata.get("iscrowd", 0)),
    }


def _bbox_to_yolo(annotation: AnnotationRecord, image: ImageRecord, category_index: int) -> str:
    """Convert one annotation to the Ultralytics YOLO txt format."""

    x, y, width, height = annotation.bbox
    image_width = max(image.width or 1, 1)
    image_height = max(image.height or 1, 1)
    center_x = (x + width / 2.0) / image_width
    center_y = (y + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return (
        f"{category_index} "
        f"{center_x:.8f} {center_y:.8f} {norm_width:.8f} {norm_height:.8f}"
    )


def _write_yolo_labels(
    label_dir: Path,
    image_ids: Sequence[str],
    annotations_by_image: Mapping[str, Sequence[AnnotationRecord]],
    images_by_id: Mapping[str, ImageRecord],
    class_names: Sequence[str],
) -> None:
    """Write YOLO labels for one split."""

    class_index = {label: index for index, label in enumerate(class_names)}
    for image_id in image_ids:
        image = images_by_id[image_id]
        label_path = label_dir / f"{Path(image.path).stem}.txt"
        lines = [
            _bbox_to_yolo(annotation, image, class_index[annotation.label])
            for annotation in annotations_by_image.get(image_id, [])
            if annotation.label in class_index
        ]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_detection_benchmark_dataset(
    *,
    images_manifest_path: Path,
    annotations_manifest_path: Path,
    splits_path: Path,
    output_dir: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> BenchmarkDatasetArtifacts:
    """Create COCO and Ultralytics views from the frozen split."""

    images = load_images_manifest(images_manifest_path)
    annotations = load_annotations(annotations_manifest_path, default_source="gt")
    splits = read_json(splits_path, default={})

    clean_directory(output_dir)

    images_by_id = {image.id: image for image in images}
    annotations_by_image = _group_annotations_by_image(annotations)
    category_by_label, categories = _build_category_maps(class_names)

    coco_root = output_dir / "coco"
    coco_annotations_dir = coco_root / "annotations"
    ultralytics_root = output_dir / "ultralytics"
    ultralytics_images_root = ultralytics_root / "images"
    ultralytics_labels_root = ultralytics_root / "labels"

    split_to_image_dir: dict[str, Path] = {}
    split_to_annotation_path: dict[str, Path] = {}
    split_to_yolo_image_dir: dict[str, Path] = {}
    split_to_yolo_label_dir: dict[str, Path] = {}
    image_numeric_id_by_id: dict[str, int] = {}
    image_summary: dict[str, Any] = {}

    running_annotation_id = 1
    running_image_id = 1

    for split_name, coco_split_name in COCO_SPLIT_NAMES.items():
        split_image_ids = list(splits.get(split_name, []))
        coco_image_dir = coco_root / coco_split_name
        yolo_image_dir = ultralytics_images_root / split_name
        yolo_label_dir = ultralytics_labels_root / split_name
        coco_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []

        for image_id in split_image_ids:
            image = images_by_id[image_id]
            numeric_image_id = running_image_id
            running_image_id += 1
            image_numeric_id_by_id[image_id] = numeric_image_id

            image_name = image.path.name
            coco_target = coco_image_dir / image_name
            yolo_target = yolo_image_dir / image_name
            _safe_symlink(image.path, coco_target)
            _safe_symlink(image.path, yolo_target)

            coco_images.append(
                {
                    "id": numeric_image_id,
                    "width": int(image.width or 0),
                    "height": int(image.height or 0),
                    "file_name": image_name,
                }
            )

            for annotation in annotations_by_image.get(image_id, []):
                if annotation.label not in category_by_label:
                    continue
                coco_annotations.append(
                    _annotation_to_coco(
                        annotation,
                        image_numeric_id=numeric_image_id,
                        annotation_numeric_id=running_annotation_id,
                        category_by_label=category_by_label,
                    )
                )
                running_annotation_id += 1

        annotation_path = coco_annotations_dir / f"instances_{coco_split_name}.json"
        write_json(
            annotation_path,
            {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            },
        )
        _write_yolo_labels(
            yolo_label_dir,
            split_image_ids,
            annotations_by_image,
            images_by_id,
            class_names,
        )

        split_to_image_dir[split_name] = coco_image_dir
        split_to_annotation_path[split_name] = annotation_path
        split_to_yolo_image_dir[split_name] = yolo_image_dir
        split_to_yolo_label_dir[split_name] = yolo_label_dir
        image_summary[split_name] = {
            "num_images": len(split_image_ids),
            "num_annotations": len(coco_annotations),
            "image_dir": coco_image_dir.as_posix(),
            "annotation_path": annotation_path.as_posix(),
        }

    ultralytics_dataset_yaml = ultralytics_root / "dataset.yaml"
    write_yaml(
        ultralytics_dataset_yaml,
        {
            "path": ultralytics_root.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: label for index, label in enumerate(class_names)},
        },
    )

    summary_path = output_dir / "summary.json"
    write_json(
        summary_path,
        {
            "class_names": list(class_names),
            "splits_path": splits_path.as_posix(),
            "images_manifest_path": images_manifest_path.as_posix(),
            "annotations_manifest_path": annotations_manifest_path.as_posix(),
            "views": {
                "coco_root": coco_root.as_posix(),
                "ultralytics_root": ultralytics_root.as_posix(),
                "ultralytics_dataset_yaml": ultralytics_dataset_yaml.as_posix(),
            },
            "splits": image_summary,
        },
    )

    return BenchmarkDatasetArtifacts(
        root_dir=output_dir,
        coco_root=coco_root,
        coco_annotations_dir=coco_annotations_dir,
        ultralytics_root=ultralytics_root,
        summary_path=summary_path,
        split_to_image_dir=split_to_image_dir,
        split_to_annotation_path=split_to_annotation_path,
        split_to_yolo_image_dir=split_to_yolo_image_dir,
        split_to_yolo_label_dir=split_to_yolo_label_dir,
        ultralytics_dataset_yaml=ultralytics_dataset_yaml,
    )
