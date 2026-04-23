"""Generate deterministic train/val/test splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.load import index_images_by_id, load_annotations, load_images_manifest  # noqa: E402
from towervision.data.splits import (  # noqa: E402
    compute_split_distribution,
    export_split_visual_samples,
    generate_official_grouped_split,
    render_split_distribution_markdown,
    save_splits,
)
from towervision.utils.io import read_yaml, write_json, write_text  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    params = read_yaml(ROOT / "params.yaml")
    split_config = read_yaml(ROOT / "configs/data/splits.yaml")
    images = load_images_manifest(_resolve_repo_path(params["paths"]["cleaned_images_manifest"]))
    annotations = load_annotations(
        _resolve_repo_path(params["paths"]["cleaned_annotations_manifest"]),
        allow_missing=True,
    )

    split_mapping, split_metadata = generate_official_grouped_split(
        images,
        train_ratio=params["data"]["train_ratio"],
        val_ratio=params["data"]["val_ratio"],
        bucket_seconds=split_config["group_bucket_seconds"],
    )
    split_path = _resolve_repo_path(params["paths"]["splits_path"])
    split_dir = split_path.parent
    save_splits(split_path, split_mapping)

    distribution = compute_split_distribution(
        split_mapping,
        annotations,
        image_to_group_id=split_metadata["image_to_group_id"],
    )
    sample_exports = export_split_visual_samples(
        index_images_by_id(images),
        annotations,
        split_mapping,
        output_dir=split_dir / "samples",
        samples_per_split=split_config["samples_per_split"],
    )

    split_metadata.update(
        {
            "dataset_name": params["dataset"]["name"],
            "dataset_version": params["dataset"]["version"],
            "split_name": split_config["split_name"],
            "distribution": distribution,
            "sample_exports": sample_exports,
            "uncertainties": split_config["uncertainties"],
        }
    )
    write_json(split_dir / "split_metadata.json", split_metadata)
    write_json(split_dir / "split_distribution.json", distribution)
    write_text(
        split_dir / "split_distribution.md",
        render_split_distribution_markdown(
            distribution,
            protocol_version=split_metadata["protocol_version"],
            group_strategy=split_metadata["group_strategy"],
            bucket_seconds=split_metadata["bucket_seconds"],
        ),
    )

    print(f"split_name: {split_metadata['split_name']}")
    print(f"protocol_version: {split_metadata['protocol_version']}")
    print(f"group_strategy: {split_metadata['group_strategy']}")
    print(f"bucket_seconds: {split_metadata['bucket_seconds']}")
    for split_name, stats in distribution.items():
        class_counts = stats["class_counts"]
        print(
            f"- {split_name}: images={stats['num_images']}, annotations={stats['num_annotations']}, "
            f"torre={class_counts.get('torre', 0)}, isoladores={class_counts.get('isoladores', 0)}, "
            f"groups={stats['num_groups']}"
        )
    print(f"- splits: {split_path}")
    print(f"- metadata: {split_dir / 'split_metadata.json'}")
    print(f"- distribution: {split_dir / 'split_distribution.md'}")
    print(f"- samples: {split_dir / 'samples'}")


if __name__ == "__main__":
    main()
