"""Utilities for controlled synthetic anomaly packs."""

from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from towervision.data.load import AnnotationRecord, ImageRecord
from towervision.pipelines.crop_from_gt import crop_bbox
from towervision.utils.io import ensure_dir, read_json, write_text, write_yaml
from towervision.utils.viz import draw_mask_overlay, render_contact_sheet

DEFAULT_GENERATORS = ("chatgpt", "gemini")
DEFAULT_SOURCE_SPLITS = ("val", "test")
PROMPT_SPECS = (
    {"slug": "crack", "anomaly_type": "crack", "severity": "moderate"},
    {
        "slug": "partial_chipping",
        "anomaly_type": "partial chipping",
        "severity": "moderate",
    },
    {"slug": "burn_mark", "anomaly_type": "burn mark", "severity": "moderate"},
    {
        "slug": "severe_contamination",
        "anomaly_type": "severe contamination",
        "severity": "moderate",
    },
    {
        "slug": "localized_surface_damage",
        "anomaly_type": "localized surface damage",
        "severity": "moderate",
    },
)
RECORD_FIELDS = [
    "record_id",
    "pair_id",
    "source_image_id",
    "source_image_path",
    "source_crop_path",
    "source_split",
    "generator_family",
    "generator_model",
    "anomaly_scope",
    "anomaly_type",
    "severity",
    "output_image_path",
    "mask_path",
    "prompt_path",
    "accepted_for_benchmark",
    "notes",
]
SOURCE_CANDIDATE_FIELDS = [
    "source_crop_id",
    "source_split",
    "source_image_id",
    "source_image_path",
    "annotation_id",
    "label",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "crop_width",
    "crop_height",
    "padding",
    "source_crop_path",
]
SOURCE_SHORTLIST_FIELDS = SOURCE_CANDIDATE_FIELDS + [
    "shortlist_rank",
    "selection_reason",
]


@dataclass(slots=True)
class SyntheticPackPaths:
    """Paths used by one synthetic anomaly pack."""

    root_dir: Path
    manifest_path: Path
    records_path: Path
    readme_path: Path
    generated_root: Path
    prompts_root: Path
    masks_root: Path
    generated_dirs: dict[str, Path]
    prompt_dirs: dict[str, Path]
    source_crops_root: Path
    source_crop_dirs: dict[str, Path]
    source_candidates_path: Path
    source_shortlist_path: Path
    source_shortlist_bundle_dir: Path


def build_synthetic_pack_paths(
    project_root: Path,
    *,
    dataset_name: str,
    dataset_version: str,
    pack_name: str,
    generators: Iterable[str] = DEFAULT_GENERATORS,
    source_splits: Iterable[str] = DEFAULT_SOURCE_SPLITS,
) -> SyntheticPackPaths:
    """Build canonical paths for one controlled synthetic anomaly pack."""

    root_dir = (
        project_root
        / "data"
        / "synthetic"
        / dataset_name
        / dataset_version
        / pack_name
    )
    generated_root = root_dir / "generated"
    prompts_root = root_dir / "prompts"
    masks_root = root_dir / "masks"
    source_crops_root = root_dir / "source_crops"
    generator_names = [str(generator) for generator in generators]
    source_split_names = [str(split_name) for split_name in source_splits]
    return SyntheticPackPaths(
        root_dir=root_dir,
        manifest_path=root_dir / "manifest.yaml",
        records_path=root_dir / "records.csv",
        readme_path=root_dir / "README.md",
        generated_root=generated_root,
        prompts_root=prompts_root,
        masks_root=masks_root,
        generated_dirs={name: generated_root / name for name in generator_names},
        prompt_dirs={name: prompts_root / name for name in generator_names},
        source_crops_root=source_crops_root,
        source_crop_dirs={name: source_crops_root / name for name in source_split_names},
        source_candidates_path=root_dir / "source_candidates.csv",
        source_shortlist_path=root_dir / "source_shortlist.csv",
        source_shortlist_bundle_dir=root_dir / "source_shortlist_bundle",
    )


def initialize_controlled_synthetic_pack(
    project_root: Path,
    *,
    dataset_name: str,
    dataset_version: str,
    pack_name: str = "anomaly_controlled_v1",
    raw_dataset_root: Path,
    generators: Iterable[str] = DEFAULT_GENERATORS,
    images_per_generator: int = 5,
    initial_roi_scope: str = "isoladores",
    future_roi_scopes: Iterable[str] = ("torre",),
    source_splits: Iterable[str] = DEFAULT_SOURCE_SPLITS,
) -> SyntheticPackPaths:
    """Create the directory layout and template files for a synthetic anomaly pack."""

    paths = build_synthetic_pack_paths(
        project_root,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        pack_name=pack_name,
        generators=generators,
        source_splits=source_splits,
    )
    ensure_dir(paths.root_dir)
    ensure_dir(paths.generated_root)
    ensure_dir(paths.prompts_root)
    ensure_dir(paths.masks_root)
    ensure_dir(paths.source_crops_root)
    ensure_dir(paths.source_shortlist_bundle_dir)
    for generator_dir in paths.generated_dirs.values():
        ensure_dir(generator_dir)
    for prompt_dir in paths.prompt_dirs.values():
        ensure_dir(prompt_dir)
    for source_crop_dir in paths.source_crop_dirs.values():
        ensure_dir(source_crop_dir)

    generator_list = [name for name in paths.generated_dirs]
    manifest = {
        "pack_name": pack_name,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "task": "controlled_synthetic_anomaly_benchmark",
        "source_dataset_root": raw_dataset_root.as_posix(),
        "protocol": {
            "generators": generator_list,
            "images_per_generator": images_per_generator,
            "total_target_images": images_per_generator * len(generator_list),
            "initial_roi_scope": [initial_roi_scope],
            "future_roi_scopes": list(future_roi_scopes),
            "source_splits": list(paths.source_crop_dirs),
        },
        "rules": [
            "Do not move or overwrite the raw dataset.",
            "Do not mix synthetic files into data/raw.",
            "Reference source images in records.csv instead of copying the originals.",
            "Store every generated image with prompt and optional mask for traceability.",
        ],
    }
    write_yaml(paths.manifest_path, manifest)
    write_records_template(paths.records_path)
    write_source_csv(paths.source_candidates_path, SOURCE_CANDIDATE_FIELDS, [])
    write_source_csv(paths.source_shortlist_path, SOURCE_SHORTLIST_FIELDS, [])
    write_text(paths.readme_path, render_synthetic_pack_readme(paths, manifest))
    return paths


def write_records_template(path: Path) -> None:
    """Write a stable CSV template for synthetic generation records."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()


def write_source_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a stable CSV file for source crops metadata."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_token(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def export_synthetic_source_crops(
    paths: SyntheticPackPaths,
    images_by_id: Mapping[str, ImageRecord],
    annotations: Sequence[AnnotationRecord],
    split_mapping: Mapping[str, Sequence[str]],
    *,
    label: str = "isoladores",
    allowed_splits: Iterable[str] = DEFAULT_SOURCE_SPLITS,
    padding: int = 64,
    shortlist_per_split: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Export source crops from GT annotations for synthetic anomaly generation."""

    allowed_split_names = [str(split_name) for split_name in allowed_splits]
    image_to_split = {
        image_id: split_name
        for split_name in allowed_split_names
        for image_id in split_mapping.get(split_name, [])
    }

    for split_name in allowed_split_names:
        source_dir = ensure_dir(paths.source_crop_dirs[split_name])
        for existing_file in source_dir.glob("*.png"):
            existing_file.unlink()

    candidates: list[dict[str, Any]] = []
    grouped_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for annotation in annotations:
        if annotation.label != label:
            continue
        split_name = image_to_split.get(annotation.image_id)
        if split_name is None:
            continue
        image_record = images_by_id.get(annotation.image_id)
        if image_record is None or not image_record.path.exists():
            continue

        with Image.open(image_record.path) as image:
            crop = crop_bbox(image, annotation.bbox, padding=padding)
            crop_width, crop_height = crop.size
            crop_name = (
                f"{_safe_token(annotation.image_id)}__{_safe_token(annotation.id)}__"
                f"{_safe_token(annotation.label)}.png"
            )
            crop_path = paths.source_crop_dirs[split_name] / crop_name
            crop.save(crop_path)

        x, y, bbox_width, bbox_height = annotation.bbox
        row = {
            "source_crop_id": f"{annotation.image_id}__{annotation.id}",
            "source_split": split_name,
            "source_image_id": annotation.image_id,
            "source_image_path": image_record.path.as_posix(),
            "annotation_id": annotation.id,
            "label": annotation.label,
            "bbox_x": round(x, 2),
            "bbox_y": round(y, 2),
            "bbox_width": round(bbox_width, 2),
            "bbox_height": round(bbox_height, 2),
            "bbox_area": round(bbox_width * bbox_height, 2),
            "crop_width": crop_width,
            "crop_height": crop_height,
            "padding": padding,
            "source_crop_path": crop_path.as_posix(),
        }
        candidates.append(row)
        grouped_by_split[split_name].append(row)

    candidates.sort(
        key=lambda row: (
            row["source_split"],
            row["source_image_id"],
            row["annotation_id"],
        )
    )
    write_source_csv(paths.source_candidates_path, SOURCE_CANDIDATE_FIELDS, candidates)

    shortlist: list[dict[str, Any]] = []
    for split_name in allowed_split_names:
        best_per_image: dict[str, dict[str, Any]] = {}
        for row in grouped_by_split.get(split_name, []):
            previous = best_per_image.get(row["source_image_id"])
            if previous is None or (
                row["bbox_area"],
                row["crop_height"],
                row["crop_width"],
            ) > (
                previous["bbox_area"],
                previous["crop_height"],
                previous["crop_width"],
            ):
                best_per_image[row["source_image_id"]] = row

        ordered_rows = sorted(
            best_per_image.values(),
            key=lambda row: row["source_image_id"],
        )
        if shortlist_per_split <= 0:
            split_shortlist: list[dict[str, Any]] = []
        else:
            bin_size = max(1, math.ceil(len(ordered_rows) / shortlist_per_split))
            split_shortlist = []
            for start in range(0, len(ordered_rows), bin_size):
                chunk = ordered_rows[start : start + bin_size]
                chosen = max(
                    chunk,
                    key=lambda row: (
                        row["bbox_area"],
                        row["crop_height"],
                        row["crop_width"],
                    ),
                )
                split_shortlist.append(chosen)
            split_shortlist = split_shortlist[:shortlist_per_split]

        for rank, row in enumerate(split_shortlist, start=1):
            shortlist.append(
                {
                    **row,
                    "shortlist_rank": rank,
                    "selection_reason": (
                        "largest isolador bbox per image; selected by temporal bin and bbox_area"
                    ),
                }
            )

    shortlist.sort(key=lambda row: (row["source_split"], row["shortlist_rank"]))
    write_source_csv(paths.source_shortlist_path, SOURCE_SHORTLIST_FIELDS, shortlist)
    return candidates, shortlist


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dict rows."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def materialize_shortlist_bundle(paths: SyntheticPackPaths) -> list[dict[str, str]]:
    """Copy shortlist source crops into one handoff folder for external generation."""

    ensure_dir(paths.source_shortlist_bundle_dir)
    for existing_file in paths.source_shortlist_bundle_dir.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()

    shortlist_rows = read_csv_rows(paths.source_shortlist_path)
    copied_rows: list[dict[str, str]] = []
    for row in shortlist_rows:
        source_crop_path = Path(row["source_crop_path"])
        if not source_crop_path.exists():
            continue
        split_name = row["source_split"]
        rank = row["shortlist_rank"]
        image_id = _safe_token(row["source_image_id"])
        annotation_id = _safe_token(row["annotation_id"])
        target_name = f"{split_name}_{rank}_{image_id}__{annotation_id}.png"
        target_path = paths.source_shortlist_bundle_dir / target_name
        shutil.copy2(source_crop_path, target_path)
        copied_rows.append(
            {
                **row,
                "bundle_path": target_path.as_posix(),
            }
        )
    return copied_rows


def _generator_from_name(file_name: str) -> str | None:
    stem = Path(file_name).stem.lower()
    if stem.endswith("_gemini"):
        return "gemini"
    if stem.endswith("_chatgpt") or stem.endswith("_gpt"):
        return "chatgpt"
    return None


def _pair_id_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    for suffix in ("_gemini", "_chatgpt", "_gpt"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _remove_note_token(notes: str, token: str) -> str:
    tokens = [item for item in notes.split(";") if item and item != token]
    return ";".join(tokens)


def _append_note_token(notes: str, token: str) -> str:
    tokens = [item for item in notes.split(";") if item]
    if token not in tokens:
        tokens.append(token)
    return ";".join(tokens)


def _normalize_roboflow_export_name(file_name: str, extra_name: str | None = None) -> str:
    if extra_name:
        return Path(extra_name).name

    normalized_name = Path(file_name).name
    if ".rf." in normalized_name:
        normalized_name = normalized_name.split(".rf.", maxsplit=1)[0]
    stem = Path(normalized_name).stem
    lower_stem = stem.lower()
    for extension in ("png", "jpg", "jpeg"):
        suffix = f"_{extension}"
        if lower_stem.endswith(suffix):
            return f"{stem[: -len(suffix)]}.{extension}"
    return normalized_name


def _decode_coco_rle_counts(encoded_counts: str) -> list[int]:
    counts: list[int] = []
    pointer = 0
    while pointer < len(encoded_counts):
        value = 0
        shift_index = 0
        has_more = True
        while has_more:
            char_code = ord(encoded_counts[pointer]) - 48
            value |= (char_code & 0x1F) << (5 * shift_index)
            has_more = (char_code & 0x20) != 0
            pointer += 1
            shift_index += 1
            if not has_more and (char_code & 0x10):
                value |= -1 << (5 * shift_index)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts


def _decode_coco_rle_mask(segmentation: Mapping[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in segmentation["size"])
    raw_counts = segmentation["counts"]
    counts = (
        _decode_coco_rle_counts(str(raw_counts))
        if isinstance(raw_counts, str)
        else [int(value) for value in raw_counts]
    )

    flat_mask = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    fill_value = 0
    for run_length in counts:
        if run_length < 0:
            raise ValueError(f"Invalid negative run length: {run_length}")
        if fill_value == 1 and run_length:
            flat_mask[cursor : cursor + run_length] = 1
        cursor += run_length
        fill_value = 1 - fill_value
    if cursor != flat_mask.size:
        raise ValueError(
            f"Decoded RLE length mismatch: expected {flat_mask.size}, got {cursor}"
        )
    return flat_mask.reshape((height, width), order="F")


def _decode_polygon_mask(
    polygons: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    mask_image = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask_image)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon) - 1, 2)
        ]
        drawer.polygon(points, fill=1)
    return np.asarray(mask_image, dtype=np.uint8)


def _decode_coco_segmentation_mask(
    segmentation: Any,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if isinstance(segmentation, Mapping):
        return _decode_coco_rle_mask(segmentation)
    if isinstance(segmentation, Sequence) and not isinstance(segmentation, (str, bytes)):
        polygons = [
            [float(value) for value in polygon]
            for polygon in segmentation
            if isinstance(polygon, Sequence)
        ]
        return _decode_polygon_mask(polygons, width=width, height=height)
    raise ValueError(f"Unsupported segmentation payload: {type(segmentation)!r}")


def _find_roboflow_annotation_path(export_root: Path) -> Path:
    matches = sorted(export_root.rglob("_annotations.coco.json"))
    if not matches:
        raise FileNotFoundError(
            f"No _annotations.coco.json found under {export_root.as_posix()}"
        )
    return matches[0]


def import_roboflow_segmentation_masks(
    paths: SyntheticPackPaths,
    export_root: Path,
) -> dict[str, Any]:
    """Import masks from a Roboflow COCO segmentation export into the synthetic pack."""

    records = read_csv_rows(paths.records_path)
    if not records:
        raise ValueError("records.csv is empty; synchronize generated outputs first")

    annotation_path = _find_roboflow_annotation_path(export_root)
    payload = read_json(annotation_path, default={})
    raw_images = payload.get("images", [])
    raw_annotations = payload.get("annotations", [])

    records_by_output_name = {
        Path(row["output_image_path"]).name: row
        for row in records
        if row.get("output_image_path")
    }
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in raw_annotations:
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    matched_output_names: set[str] = set()
    imported_records = 0
    unmatched_export_images = 0
    missing_annotation_images = 0

    for raw_image in raw_images:
        normalized_name = _normalize_roboflow_export_name(
            str(raw_image.get("file_name", "")),
            str(raw_image.get("extra", {}).get("name", "")) or None,
        )
        record = records_by_output_name.get(normalized_name)
        if record is None:
            unmatched_export_images += 1
            continue

        matched_output_names.add(normalized_name)
        image_annotations = annotations_by_image_id.get(int(raw_image["id"]), [])
        if not image_annotations:
            missing_annotation_images += 1
            record["notes"] = _append_note_token(
                record.get("notes", ""),
                "mask_import_missing_annotation=true",
            )
            continue

        height = int(raw_image["height"])
        width = int(raw_image["width"])
        merged_mask = np.zeros((height, width), dtype=np.uint8)
        for annotation in image_annotations:
            segmentation = annotation.get("segmentation")
            if segmentation is None:
                continue
            decoded_mask = _decode_coco_segmentation_mask(
                segmentation,
                width=width,
                height=height,
            )
            merged_mask = np.maximum(merged_mask, decoded_mask)

        if not merged_mask.any():
            missing_annotation_images += 1
            record["notes"] = _append_note_token(
                record.get("notes", ""),
                "mask_import_empty_mask=true",
            )
            continue

        split_name = record.get("source_split") or "unknown"
        mask_dir = ensure_dir(paths.masks_root / split_name)
        mask_path = mask_dir / f"{record['record_id']}__mask.png"
        Image.fromarray((merged_mask * 255).astype(np.uint8), mode="L").save(mask_path)

        record["mask_path"] = mask_path.as_posix()
        updated_notes = _remove_note_token(record.get("notes", ""), "pending_mask_annotation")
        updated_notes = _append_note_token(updated_notes, "mask_imported_from_roboflow=true")
        updated_notes = _append_note_token(updated_notes, f"mask_source_split={split_name}")
        record["notes"] = updated_notes
        imported_records += 1

    for row in records:
        output_name = Path(row["output_image_path"]).name if row.get("output_image_path") else ""
        if row.get("mask_path"):
            continue
        if output_name and output_name not in matched_output_names:
            row["notes"] = _append_note_token(
                row.get("notes", ""),
                "mask_import_missing_from_export=true",
            )

    write_source_csv(paths.records_path, RECORD_FIELDS, records)
    return {
        "annotation_path": annotation_path.as_posix(),
        "export_root": export_root.as_posix(),
        "export_image_count": len(raw_images),
        "export_annotation_count": len(raw_annotations),
        "matched_record_count": len(matched_output_names),
        "imported_mask_count": imported_records,
        "missing_annotation_count": missing_annotation_images,
        "unmatched_export_image_count": unmatched_export_images,
        "records_path": paths.records_path.as_posix(),
        "masks_root": paths.masks_root.as_posix(),
    }


def render_synthetic_mask_overlays(
    paths: SyntheticPackPaths,
    *,
    output_root: Path,
) -> list[dict[str, str]]:
    """Render overlay images for synthetic masks using records.csv as index."""

    records = read_csv_rows(paths.records_path)
    rendered: list[dict[str, str]] = []
    for row in records:
        output_image_path = Path(row.get("output_image_path", ""))
        mask_path = Path(row.get("mask_path", ""))
        if not output_image_path.exists() or not mask_path.exists():
            continue

        split_name = row.get("source_split") or "unknown"
        generator_name = row.get("generator_family") or "unknown"
        overlay_dir = ensure_dir(output_root / split_name / generator_name)
        overlay_path = overlay_dir / f"{row['record_id']}__overlay.png"
        draw_mask_overlay(
            output_image_path,
            mask_path,
            output_path=overlay_path,
        )
        rendered.append(
            {
                "record_id": row["record_id"],
                "source_split": split_name,
                "generator_family": generator_name,
                "output_image_path": output_image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "overlay_path": overlay_path.as_posix(),
            }
        )
    return rendered


def render_synthetic_overlay_contact_sheet(
    paths: SyntheticPackPaths,
    *,
    overlay_root: Path,
    output_path: Path,
    columns: int = 4,
) -> dict[str, Any]:
    """Render one contact sheet covering all overlay images in records.csv order."""

    records = read_csv_rows(paths.records_path)
    items: list[tuple[Path, str]] = []
    for row in records:
        split_name = row.get("source_split") or "unknown"
        generator_name = row.get("generator_family") or "unknown"
        overlay_path = (
            overlay_root
            / split_name
            / generator_name
            / f"{row['record_id']}__overlay.png"
        )
        if not overlay_path.exists():
            continue
        suggested_severity = ""
        for token in row.get("notes", "").split(";"):
            if token.startswith("review_suggested_severity="):
                suggested_severity = token.split("=", maxsplit=1)[1]
                break
        label = "\n".join(
            [
                row["record_id"],
                f"{split_name} | {generator_name}",
                suggested_severity or row.get("severity", ""),
            ]
        )
        items.append((overlay_path, label))

    contact_sheet_path = render_contact_sheet(
        items,
        output_path=output_path,
        columns=columns,
        title="Synthetic Anomaly Mask Review",
    )
    return {
        "contact_sheet_path": contact_sheet_path.as_posix(),
        "overlay_item_count": len(items),
        "columns": columns,
    }


def accept_synthetic_records_for_benchmark(paths: SyntheticPackPaths) -> dict[str, Any]:
    """Mark synthetic records as accepted when they are visually valid and have masks."""

    records = read_csv_rows(paths.records_path)
    accepted_count = 0
    rejected_count = 0
    for row in records:
        notes = row.get("notes", "")
        is_valid = "review_valid=true" in notes
        has_mask = bool(row.get("mask_path")) and Path(row["mask_path"]).exists()
        if is_valid and has_mask:
            row["accepted_for_benchmark"] = "true"
            row["notes"] = _append_note_token(notes, "accepted_for_benchmark_by_curated_review=true")
            accepted_count += 1
        else:
            row["accepted_for_benchmark"] = "false"
            row["notes"] = _remove_note_token(
                notes,
                "accepted_for_benchmark_by_curated_review=true",
            )
            rejected_count += 1

    write_source_csv(paths.records_path, RECORD_FIELDS, records)
    return {
        "records_path": paths.records_path.as_posix(),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    }


def render_anomaly_prompt(anomaly_type: str, severity: str) -> str:
    """Render the standard prompt used for one synthetic anomaly edit."""

    return (
        "You are editing a real inspection photo crop of a power-line insulator.\n\n"
        "Task:\n"
        "Insert exactly one realistic visual anomaly into the insulator only.\n\n"
        "Primary goal:\n"
        "Create a photorealistic edited version of the input image that still looks like a real "
        f"inspection crop, while adding one localized anomaly of type {anomaly_type} with severity "
        f"{severity}.\n\n"
        "Hard constraints:\n"
        "- Keep the same framing, aspect ratio, resolution, perspective, and camera viewpoint.\n"
        "- Preserve the same background, lighting, shadows, color balance, and overall photographic style.\n"
        "- Preserve the global geometry and identity of the insulator.\n"
        "- Edit only a small localized region of the insulator.\n"
        "- Do not add any new objects outside the insulator.\n"
        "- Do not change the tower, cables, sky, background, or scene composition.\n"
        "- Do not create multiple defects.\n"
        "- Do not turn the image into an illustration, CGI render, painting, or synthetic-looking artwork.\n"
        "- Do not degrade the whole image.\n"
        "- The result must remain visually plausible for a technical inspection scenario.\n\n"
        "Anomaly specification:\n"
        "- anomaly scope: insulator only\n"
        f"- anomaly type: {anomaly_type}\n"
        f"- severity: {severity}\n\n"
        "Desired behavior:\n"
        "- The anomaly must be clearly visible but still plausible.\n"
        "- The anomaly must affect only one limited area of the insulator.\n"
        "- The rest of the insulator should remain intact.\n"
        "- Material appearance must remain coherent with the apparent insulator material in the image.\n"
        "- Keep texture realism and local consistency.\n\n"
        "Negative constraints:\n"
        "Do not change the background, tower, cables, sky, framing, aspect ratio, camera angle, "
        "lighting setup, or overall scene composition. Do not add text, watermark, logo, extra "
        "objects, multiple anomalies, unrealistic textures, painterly style, CGI appearance, or "
        "global image degradation.\n\n"
        "Output:\n"
        "Return one edited image only.\n"
    )


def materialize_prompt_library(paths: SyntheticPackPaths) -> dict[tuple[str, str], str]:
    """Create reusable prompt files for each generator and prompt slot."""

    prompt_path_index: dict[tuple[str, str], str] = {}
    for generator_name, prompt_dir in paths.prompt_dirs.items():
        for existing_file in prompt_dir.glob("*.md"):
            existing_file.unlink()
        for index, spec in enumerate(PROMPT_SPECS, start=1):
            file_name = f"{index:02d}_{spec['slug']}.md"
            prompt_path = prompt_dir / file_name
            write_text(
                prompt_path,
                render_anomaly_prompt(
                    anomaly_type=str(spec["anomaly_type"]),
                    severity=str(spec["severity"]),
                ),
            )
            prompt_path_index[(generator_name, str(spec["slug"]))] = prompt_path.as_posix()
    return prompt_path_index


def build_prompt_schedule_from_bundle(paths: SyntheticPackPaths) -> dict[str, dict[str, str]]:
    """Assign prompt specs in order of the shortlist bundle, cycling after five prompts."""

    bundle_files = sorted(paths.source_shortlist_bundle_dir.glob("*.png"))
    schedule: dict[str, dict[str, str]] = {}
    for index, bundle_path in enumerate(bundle_files):
        pair_id = Path(bundle_path).stem
        spec = PROMPT_SPECS[index % len(PROMPT_SPECS)]
        schedule[pair_id] = {
            "slug": str(spec["slug"]),
            "anomaly_type": str(spec["anomaly_type"]),
            "severity": str(spec["severity"]),
        }
    return schedule


def build_shortlist_pair_index(shortlist_rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Index shortlist rows by stable pair identifier."""

    index: dict[str, dict[str, str]] = {}
    for row in shortlist_rows:
        pair_id = (
            f"{row['source_split']}_{row['shortlist_rank']}_"
            f"{row['source_image_id']}__{row['annotation_id']}"
        )
        index[pair_id] = dict(row)
    return index


def sync_records_from_generated_outputs(paths: SyntheticPackPaths) -> list[dict[str, str]]:
    """Populate records.csv from generated outputs without inventing missing annotations."""

    shortlist_rows = read_csv_rows(paths.source_shortlist_path)
    shortlist_index = build_shortlist_pair_index(shortlist_rows)
    prompt_path_index = materialize_prompt_library(paths)
    prompt_schedule = build_prompt_schedule_from_bundle(paths)
    records: list[dict[str, str]] = []

    generated_files = sorted(
        path
        for generator_dir in paths.generated_dirs.values()
        for path in generator_dir.glob("*.png")
    )
    for generated_path in generated_files:
        generator_family = _generator_from_name(generated_path.name)
        if generator_family is None:
            continue
        pair_id = _pair_id_from_name(generated_path.name)
        source_row = shortlist_index.get(pair_id)
        if source_row is None:
            continue
        prompt_spec = prompt_schedule.get(pair_id, {})
        records.append(
            {
                "record_id": f"{pair_id}__{generator_family}",
                "pair_id": pair_id,
                "source_image_id": source_row["source_image_id"],
                "source_image_path": source_row["source_image_path"],
                "source_crop_path": source_row["source_crop_path"],
                "source_split": source_row["source_split"],
                "generator_family": generator_family,
                "generator_model": "",
                "anomaly_scope": "isoladores",
                "anomaly_type": str(prompt_spec.get("anomaly_type", "")),
                "severity": str(prompt_spec.get("severity", "")),
                "output_image_path": generated_path.as_posix(),
                "mask_path": "",
                "prompt_path": prompt_path_index.get(
                    (generator_family, str(prompt_spec.get("slug", ""))),
                    "",
                ),
                "accepted_for_benchmark": "false",
                "notes": "pending_mask_annotation",
            }
        )

    records.sort(key=lambda row: (row["pair_id"], row["generator_family"]))
    write_source_csv(paths.records_path, RECORD_FIELDS, records)
    return records


def render_synthetic_pack_readme(
    paths: SyntheticPackPaths,
    manifest: dict[str, object],
) -> str:
    """Render a short README for the controlled synthetic pack."""

    generator_lines = "\n".join(
        f"- `{name}`: `{directory.relative_to(paths.root_dir).as_posix()}/`"
        for name, directory in paths.generated_dirs.items()
    )
    prompt_lines = "\n".join(
        f"- `{name}`: `{directory.relative_to(paths.root_dir).as_posix()}/`"
        for name, directory in paths.prompt_dirs.items()
    )
    source_lines = "\n".join(
        f"- `{name}`: `{directory.relative_to(paths.root_dir).as_posix()}/`"
        for name, directory in paths.source_crop_dirs.items()
    )
    rules = manifest.get("rules", [])
    rule_lines = "\n".join(f"- {rule}" for rule in rules if isinstance(rule, str))
    return (
        f"# {manifest['pack_name']}\n\n"
        "Pacote controlado de anomalias sintéticas para benchmark não supervisionado.\n\n"
        "## Estrutura\n\n"
        f"- manifest: `{paths.manifest_path.name}`\n"
        f"- registros: `{paths.records_path.name}`\n"
        f"- imagens geradas:\n{generator_lines}\n"
        f"- prompts:\n{prompt_lines}\n"
        f"- máscaras opcionais: `{paths.masks_root.relative_to(paths.root_dir).as_posix()}/`\n\n"
        f"- source crops:\n{source_lines}\n"
        f"- candidatos exportados: `{paths.source_candidates_path.name}`\n"
        f"- shortlist recomendada: `{paths.source_shortlist_path.name}`\n\n"
        f"- pasta pronta para handoff: `{paths.source_shortlist_bundle_dir.relative_to(paths.root_dir).as_posix()}/`\n\n"
        "## Regras\n\n"
        f"{rule_lines}\n\n"
        "## Uso esperado\n\n"
        "- cada linha em `records.csv` representa uma imagem sintética aceita ou em revisão;\n"
        "- `source_candidates.csv` guarda todos os crops exportados de `val` e `test`;\n"
        "- `source_shortlist.csv` guarda uma shortlist inicial para geração controlada;\n"
        "- `source_image_path` deve apontar para a imagem real original no dataset versionado;\n"
        "- `source_crop_path` deve apontar para um crop em `source_crops/<split>/`;\n"
        "- `pair_id` deve ser igual entre ChatGPT e Gemini quando os dois usarem o mesmo crop base;\n"
        "- `severity` deve registrar o nível da anomalia simulada;\n"
        "- `output_image_path` deve apontar para um arquivo dentro de `generated/<generator>/`;\n"
        "- `prompt_path` deve apontar para um `.md` dentro de `prompts/<generator>/`;\n"
        "- `mask_path` é opcional e deve ficar em `masks/` quando existir.\n"
    )
