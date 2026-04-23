"""Materialize the anomaly benchmark dataset views."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from towervision.anomaly.benchmark_types import (
    ANOMALY_BENCHMARK_ROW_FIELDS,
    AnomalyBenchmarkDatasetArtifacts,
    AnomalyBenchmarkDatasetRow,
)
from towervision.data.load import AnnotationRecord, ImageRecord, load_annotations, load_images_manifest
from towervision.data.synthetic import read_csv_rows
from towervision.pipelines.crop_from_gt import crop_bbox
from towervision.utils.io import clean_directory, ensure_dir, read_json, write_json


def _safe_token(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _safe_symlink(source: Path, target: Path) -> None:
    """Create or replace a symlink."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def _write_manifest_csv(path: Path, rows: Sequence[AnomalyBenchmarkDatasetRow]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANOMALY_BENCHMARK_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def read_dataset_manifest(path: Path) -> list[AnomalyBenchmarkDatasetRow]:
    """Load one anomaly dataset manifest CSV."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [AnomalyBenchmarkDatasetRow.from_dict(row) for row in csv.DictReader(handle)]


def _resolve_crop_padding(synthetic_records_path: Path) -> int:
    """Infer the canonical crop padding from the synthetic pack metadata."""

    source_candidates_path = synthetic_records_path.parent / "source_candidates.csv"
    rows = read_csv_rows(source_candidates_path)
    paddings = {
        int(float(row["padding"]))
        for row in rows
        if row.get("padding") not in ("", None)
    }
    if not paddings:
        return 0
    if len(paddings) > 1:
        raise ValueError(f"ambiguous crop padding values in {source_candidates_path}: {sorted(paddings)}")
    return next(iter(paddings))


def _build_source_crop_index(synthetic_records_path: Path) -> dict[tuple[str, str], Path]:
    """Index reusable source crops exported for the synthetic pack."""

    source_candidates_path = synthetic_records_path.parent / "source_candidates.csv"
    index: dict[tuple[str, str], Path] = {}
    for row in read_csv_rows(source_candidates_path):
        key = (str(row["source_image_id"]), str(row["annotation_id"]))
        crop_path = Path(str(row["source_crop_path"]))
        if crop_path.exists():
            index[key] = crop_path
    return index


def _materialize_normal_rows(
    *,
    split_name: str,
    image_ids: Sequence[str],
    images_by_id: Mapping[str, ImageRecord],
    annotations: Sequence[AnnotationRecord],
    output_dir: Path,
    padding: int,
    reusable_source_crops: Mapping[tuple[str, str], Path],
) -> list[AnomalyBenchmarkDatasetRow]:
    """Create the normal ROI rows for one split."""

    rows: list[AnomalyBenchmarkDatasetRow] = []
    annotations_by_image: dict[str, list[AnnotationRecord]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

    normal_dir = ensure_dir(output_dir / "normal")
    for image_id in image_ids:
        image_record = images_by_id.get(image_id)
        if image_record is None or not image_record.path.exists():
            continue
        image_annotations = [item for item in annotations_by_image.get(image_id, []) if item.label == "isoladores"]
        if not image_annotations:
            continue
        with Image.open(image_record.path) as image:
            for annotation in image_annotations:
                crop_name = (
                    f"{_safe_token(annotation.image_id)}__{_safe_token(annotation.id)}__"
                    f"{_safe_token(annotation.label)}.png"
                )
                target_path = normal_dir / crop_name
                existing_source_crop = reusable_source_crops.get((annotation.image_id, annotation.id))
                if existing_source_crop is not None and existing_source_crop.exists():
                    _safe_symlink(existing_source_crop, target_path)
                    source_crop_path = existing_source_crop.as_posix()
                else:
                    crop = crop_bbox(image, annotation.bbox, padding=padding)
                    crop.save(target_path)
                    source_crop_path = target_path.as_posix()
                rows.append(
                    AnomalyBenchmarkDatasetRow(
                        roi_id=f"{split_name}__{annotation.image_id}__{annotation.id}",
                        record_id="",
                        pair_id="",
                        image_id=annotation.image_id,
                        source_image_path=image_record.path.as_posix(),
                        source_crop_path=source_crop_path,
                        crop_path=target_path.as_posix(),
                        mask_path="",
                        split=split_name,
                        label=0,
                        source_kind="normal_gt",
                    )
                )
    return rows


def _materialize_anomaly_rows(
    *,
    split_name: str,
    synthetic_records: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> list[AnomalyBenchmarkDatasetRow]:
    """Create the accepted synthetic anomaly rows for one split."""

    anomaly_dir = ensure_dir(output_dir / "anomaly")
    masks_dir = ensure_dir(output_dir / "masks")
    rows: list[AnomalyBenchmarkDatasetRow] = []
    for record in synthetic_records:
        output_image_path = Path(str(record["output_image_path"]))
        if not output_image_path.exists():
            continue
        target_image_path = anomaly_dir / output_image_path.name
        _safe_symlink(output_image_path, target_image_path)

        mask_path = ""
        raw_mask_path = str(record.get("mask_path", ""))
        if raw_mask_path:
            source_mask_path = Path(raw_mask_path)
            if source_mask_path.exists():
                target_mask_path = masks_dir / output_image_path.name
                _safe_symlink(source_mask_path, target_mask_path)
                mask_path = target_mask_path.as_posix()

        rows.append(
            AnomalyBenchmarkDatasetRow(
                roi_id=str(record["record_id"]),
                record_id=str(record["record_id"]),
                pair_id=str(record.get("pair_id", "")),
                image_id=str(record["source_image_id"]),
                source_image_path=str(record.get("source_image_path", "")),
                source_crop_path=str(record.get("source_crop_path", "")),
                crop_path=target_image_path.as_posix(),
                mask_path=mask_path,
                split=split_name,
                label=1,
                source_kind="synthetic_anomaly",
                generator_family=str(record.get("generator_family", "")),
                anomaly_type=str(record.get("anomaly_type", "")),
                severity=str(record.get("severity", "")),
            )
        )
    return rows


def materialize_anomaly_benchmark_dataset(
    *,
    images_manifest_path: Path,
    annotations_manifest_path: Path,
    splits_path: Path,
    synthetic_records_path: Path,
    output_dir: Path,
    roi_label: str = "isoladores",
) -> AnomalyBenchmarkDatasetArtifacts:
    """Materialize train/val/test ROI views for the anomaly benchmark."""

    images = load_images_manifest(images_manifest_path)
    annotations = [
        annotation
        for annotation in load_annotations(annotations_manifest_path, default_source="gt")
        if annotation.label == roi_label
    ]
    splits = read_json(splits_path, default={})
    records = [
        row
        for row in read_csv_rows(synthetic_records_path)
        if row.get("accepted_for_benchmark", "").lower() == "true"
    ]

    clean_directory(output_dir)
    images_by_id = {image.id: image for image in images}
    crop_padding = _resolve_crop_padding(synthetic_records_path)
    reusable_source_crops = _build_source_crop_index(synthetic_records_path)

    split_to_root_dir: dict[str, Path] = {}
    split_to_manifest_path: dict[str, Path] = {}
    split_to_normal_dir: dict[str, Path] = {}
    split_to_anomaly_dir: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "images_manifest_path": images_manifest_path.as_posix(),
        "annotations_manifest_path": annotations_manifest_path.as_posix(),
        "splits_path": splits_path.as_posix(),
        "synthetic_records_path": synthetic_records_path.as_posix(),
        "roi_label": roi_label,
        "crop_padding": crop_padding,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        split_root = ensure_dir(output_dir / "splits" / split_name)
        split_to_root_dir[split_name] = split_root
        split_to_normal_dir[split_name] = split_root / "normal"
        split_to_anomaly_dir[split_name] = split_root / "anomaly"

        normal_rows = _materialize_normal_rows(
            split_name=split_name,
            image_ids=list(splits.get(split_name, [])),
            images_by_id=images_by_id,
            annotations=annotations,
            output_dir=split_root,
            padding=crop_padding,
            reusable_source_crops=reusable_source_crops,
        )
        anomaly_rows = _materialize_anomaly_rows(
            split_name=split_name,
            synthetic_records=[row for row in records if row.get("source_split") == split_name],
            output_dir=split_root,
        )
        manifest_rows = sorted(
            normal_rows + anomaly_rows,
            key=lambda row: (row.label, row.image_id, row.roi_id),
        )
        manifest_path = output_dir / "manifests" / f"{split_name}.csv"
        _write_manifest_csv(manifest_path, manifest_rows)
        split_to_manifest_path[split_name] = manifest_path

        summary["splits"][split_name] = {
            "manifest_path": manifest_path.as_posix(),
            "num_rows": len(manifest_rows),
            "num_normals": len(normal_rows),
            "num_anomalies": len(anomaly_rows),
            "root_dir": split_root.as_posix(),
        }

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    return AnomalyBenchmarkDatasetArtifacts(
        root_dir=output_dir,
        summary_path=summary_path,
        split_to_root_dir=split_to_root_dir,
        split_to_manifest_path=split_to_manifest_path,
        split_to_normal_dir=split_to_normal_dir,
        split_to_anomaly_dir=split_to_anomaly_dir,
        crop_padding=crop_padding,
    )
