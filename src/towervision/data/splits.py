"""Deterministic dataset splits."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from towervision.data.load import AnnotationRecord, ImageRecord
from towervision.utils.viz import draw_labeled_boxes
from towervision.utils.io import write_json

SplitMapping = dict[str, list[str]]
SPLIT_NAMES = ("train", "val", "test")
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z]+)_(?P<date>\d{8})(?P<time>\d{6})_(?P<seq>\d+)_?(?P<suffix>.*)$"
)


@dataclass(slots=True)
class FilenameMetadata:
    """Capture metadata parsed from an image identifier."""

    prefix: str
    date: str
    time: str
    sequence: int
    suffix: str

    @property
    def captured_at(self) -> datetime:
        """Return the capture timestamp encoded in the file name."""

        return datetime.strptime(f"{self.date}{self.time}", "%Y%m%d%H%M%S")


@dataclass(slots=True)
class GroupRecord:
    """Contiguous capture group used for leakage-aware splits."""

    group_id: str
    image_ids: list[str]
    start_time: datetime
    end_time: datetime
    sequence_min: int
    sequence_max: int

    @property
    def num_images(self) -> int:
        """Number of images in the group."""

        return len(self.image_ids)


def generate_splits(
    image_ids: Iterable[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitMapping:
    """Create deterministic train/val/test splits without leakage."""

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios")

    ordered_ids = sorted(set(image_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered_ids)

    total = len(ordered_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if total > 0 and train_end == 0:
        train_end = 1
    if val_end > total:
        val_end = total

    return {
        "train": ordered_ids[:train_end],
        "val": ordered_ids[train_end:val_end],
        "test": ordered_ids[val_end:],
    }


def save_splits(path: Path, splits: SplitMapping) -> None:
    """Persist split mapping to JSON."""

    write_json(path, splits)


def parse_filename_metadata(image_id: str) -> FilenameMetadata:
    """Parse capture metadata from the normalized image identifier."""

    match = FILENAME_PATTERN.match(Path(image_id).name)
    if match is None:
        raise ValueError(f"unsupported image id format: {image_id}")
    return FilenameMetadata(
        prefix=match.group("prefix"),
        date=match.group("date"),
        time=match.group("time"),
        sequence=int(match.group("seq")),
        suffix=match.group("suffix"),
    )


def make_time_bucket_group_id(image_id: str, *, bucket_seconds: int = 30) -> str:
    """Build a deterministic group id from the capture timestamp in the file name."""

    if bucket_seconds <= 0 or 60 % bucket_seconds != 0:
        raise ValueError("bucket_seconds must be a positive divisor of 60")

    metadata = parse_filename_metadata(image_id)
    bucket_start = (metadata.captured_at.second // bucket_seconds) * bucket_seconds
    return f"{metadata.date}_{metadata.captured_at.strftime('%H%M')}_{bucket_start:02d}_{bucket_seconds:02d}s"


def build_temporal_groups(
    image_ids: Iterable[str],
    *,
    bucket_seconds: int = 30,
) -> tuple[list[GroupRecord], dict[str, str]]:
    """Group images into contiguous temporal buckets using file-name metadata."""

    grouped_ids: dict[str, list[str]] = defaultdict(list)
    metadata_by_id: dict[str, FilenameMetadata] = {}
    for image_id in image_ids:
        metadata = parse_filename_metadata(image_id)
        metadata_by_id[image_id] = metadata
        group_id = make_time_bucket_group_id(image_id, bucket_seconds=bucket_seconds)
        grouped_ids[group_id].append(image_id)

    image_to_group = {
        image_id: group_id for group_id, group_image_ids in grouped_ids.items() for image_id in group_image_ids
    }
    groups: list[GroupRecord] = []
    for group_id, group_image_ids in grouped_ids.items():
        ordered_ids = sorted(
            group_image_ids,
            key=lambda image_id: (
                metadata_by_id[image_id].captured_at,
                metadata_by_id[image_id].sequence,
            ),
        )
        start_metadata = metadata_by_id[ordered_ids[0]]
        end_metadata = metadata_by_id[ordered_ids[-1]]
        groups.append(
            GroupRecord(
                group_id=group_id,
                image_ids=ordered_ids,
                start_time=start_metadata.captured_at,
                end_time=end_metadata.captured_at + timedelta(seconds=bucket_seconds - 1),
                sequence_min=min(metadata_by_id[image_id].sequence for image_id in ordered_ids),
                sequence_max=max(metadata_by_id[image_id].sequence for image_id in ordered_ids),
            )
        )

    groups.sort(key=lambda group: (group.start_time, group.sequence_min, group.group_id))
    return groups, image_to_group


def choose_contiguous_group_boundaries(
    groups: Sequence[GroupRecord],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[int, int]:
    """Choose chronological group boundaries that best approximate target split ratios."""

    if len(groups) < 3:
        raise ValueError("at least three groups are required to create train/val/test splits")

    total_images = sum(group.num_images for group in groups)
    target_train = total_images * train_ratio
    target_val = total_images * val_ratio
    target_test = total_images * (1 - train_ratio - val_ratio)

    prefix_counts = [0]
    for group in groups:
        prefix_counts.append(prefix_counts[-1] + group.num_images)

    best_boundaries: tuple[float, float, int, int] | None = None
    for train_end in range(1, len(groups) - 1):
        for val_end in range(train_end + 1, len(groups)):
            train_images = prefix_counts[train_end]
            val_images = prefix_counts[val_end] - prefix_counts[train_end]
            test_images = total_images - prefix_counts[val_end]
            score = (
                (train_images - target_train) ** 2
                + (val_images - target_val) ** 2
                + (test_images - target_test) ** 2
            )
            max_abs_error = max(
                abs(train_images - target_train),
                abs(val_images - target_val),
                abs(test_images - target_test),
            )
            candidate = (score, max_abs_error, train_end, val_end)
            if best_boundaries is None or candidate < best_boundaries:
                best_boundaries = candidate

    if best_boundaries is None:
        raise ValueError("failed to choose split boundaries")
    _, _, train_end, val_end = best_boundaries
    return train_end, val_end


def generate_official_grouped_split(
    images: Sequence[ImageRecord],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    bucket_seconds: int = 30,
) -> tuple[SplitMapping, dict[str, Any]]:
    """Generate the official v1 split using chronological temporal groups."""

    groups, image_to_group = build_temporal_groups(
        [image.id for image in images],
        bucket_seconds=bucket_seconds,
    )
    train_end, val_end = choose_contiguous_group_boundaries(
        groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    split_mapping: SplitMapping = {split_name: [] for split_name in SPLIT_NAMES}
    split_to_group_ids: dict[str, list[str]] = {split_name: [] for split_name in SPLIT_NAMES}
    group_summaries: list[dict[str, Any]] = []

    for index, group in enumerate(groups):
        split_name = "train" if index < train_end else "val" if index < val_end else "test"
        split_mapping[split_name].extend(group.image_ids)
        split_to_group_ids[split_name].append(group.group_id)
        group_summaries.append(
            {
                "group_id": group.group_id,
                "start_time": group.start_time.isoformat(),
                "end_time": group.end_time.isoformat(),
                "sequence_min": group.sequence_min,
                "sequence_max": group.sequence_max,
                "num_images": group.num_images,
                "split": split_name,
            }
        )

    metadata = {
        "protocol_version": "official_v1",
        "group_strategy": "filename_time_bucket",
        "bucket_seconds": bucket_seconds,
        "split_strategy": "chronological_contiguous_groups",
        "train_ratio_target": train_ratio,
        "val_ratio_target": val_ratio,
        "test_ratio_target": 1 - train_ratio - val_ratio,
        "train_group_count": train_end,
        "val_group_count": val_end - train_end,
        "test_group_count": len(groups) - val_end,
        "split_to_group_ids": split_to_group_ids,
        "image_to_group_id": image_to_group,
        "groups": group_summaries,
    }
    return split_mapping, metadata


def compute_split_distribution(
    splits: Mapping[str, Sequence[str]],
    annotations: Sequence[AnnotationRecord],
    *,
    image_to_group_id: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute image and class distribution for each split."""

    image_to_split = {
        image_id: split_name
        for split_name, image_ids in splits.items()
        for image_id in image_ids
    }
    annotations_by_split: dict[str, list[AnnotationRecord]] = {split_name: [] for split_name in splits}
    for annotation in annotations:
        split_name = image_to_split.get(annotation.image_id)
        if split_name is None:
            continue
        annotations_by_split[split_name].append(annotation)

    distribution: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        image_ids = list(splits.get(split_name, []))
        split_annotations = annotations_by_split.get(split_name, [])
        class_counts = Counter(annotation.label for annotation in split_annotations)
        group_ids = (
            sorted({image_to_group_id[image_id] for image_id in image_ids})
            if image_to_group_id is not None
            else []
        )
        distribution[split_name] = {
            "num_images": len(image_ids),
            "num_annotations": len(split_annotations),
            "class_counts": dict(class_counts),
            "num_groups": len(group_ids),
            "group_ids": group_ids,
        }
    return distribution


def render_split_distribution_markdown(
    distribution: Mapping[str, Mapping[str, Any]],
    *,
    protocol_version: str,
    group_strategy: str,
    bucket_seconds: int,
) -> str:
    """Render a compact markdown report for the official split."""

    lines = [
        "# Split Distribution",
        "",
        f"- protocol_version: `{protocol_version}`",
        f"- group_strategy: `{group_strategy}`",
        f"- bucket_seconds: `{bucket_seconds}`",
        "",
        "| Split | Images | Annotations | Torre | Isoladores | Groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in SPLIT_NAMES:
        split_stats = distribution[split_name]
        class_counts = split_stats["class_counts"]
        lines.append(
            (
                f"| {split_name} | {split_stats['num_images']} | {split_stats['num_annotations']} | "
                f"{class_counts.get('torre', 0)} | {class_counts.get('isoladores', 0)} | "
                f"{split_stats['num_groups']} |"
            )
    )
    return "\n".join(lines)


def export_split_visual_samples(
    images_by_id: Mapping[str, ImageRecord],
    annotations: Sequence[AnnotationRecord],
    splits: Mapping[str, Sequence[str]],
    *,
    output_dir: Path,
    samples_per_split: int = 3,
) -> list[dict[str, str]]:
    """Export deterministic visual samples with both classes when available."""

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation.image_id].append(annotation)

    exports: list[dict[str, str]] = []
    for split_name in SPLIT_NAMES:
        candidate_ids = list(splits.get(split_name, []))
        if not candidate_ids:
            continue

        def score(image_id: str) -> tuple[int, str]:
            labels = {annotation.label for annotation in annotations_by_image.get(image_id, [])}
            has_both_classes = int({"torre", "isoladores"}.issubset(labels))
            return (-has_both_classes, image_id)

        ordered_candidates = sorted(candidate_ids, key=score)
        selected_ids = ordered_candidates[:samples_per_split]
        split_dir = output_dir / split_name

        for image_id in selected_ids:
            image_record = images_by_id.get(image_id)
            if image_record is None:
                continue
            image_annotations = annotations_by_image.get(image_id, [])
            labeled_boxes = [
                (annotation.label, annotation.bbox)
                for annotation in image_annotations
            ]
            output_path = split_dir / f"{image_id}.jpg"
            draw_labeled_boxes(
                image_record.path,
                labeled_boxes,
                output_path=output_path,
            )
            exports.append(
                {
                    "split": split_name,
                    "image_id": image_id,
                    "path": output_path.as_posix(),
                }
            )
    return exports
