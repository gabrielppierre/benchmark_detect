"""Prepare dataset manifests from raw Tower Vision inputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.load import (  # noqa: E402
    discover_images,
    load_coco_dataset,
    load_annotations,
    save_annotations,
    save_images_manifest,
)
from towervision.data.validate import build_validation_report  # noqa: E402
from towervision.utils.io import read_yaml, write_json  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    params = read_yaml(ROOT / "params.yaml")
    base_config = read_yaml(ROOT / "configs/data/base.yaml")

    paths = params["paths"]
    data_params = params["data"]
    image_extensions = data_params.get("image_extensions", base_config["image_extensions"])
    default_label = data_params.get("annotation_label", base_config["default_label"])
    annotation_format = str(
        data_params.get("annotation_format", base_config.get("annotation_format", "custom"))
    ).lower()

    if annotation_format == "coco":
        images, annotations = load_coco_dataset(
            _resolve_repo_path(paths["raw_images_dir"]),
            _resolve_repo_path(paths["raw_annotations_path"]),
        )
    else:
        images = discover_images(_resolve_repo_path(paths["raw_images_dir"]), image_extensions)
        annotations = load_annotations(
            _resolve_repo_path(paths["raw_annotations_path"]),
            default_label=default_label,
            allow_missing=True,
        )

    save_images_manifest(_resolve_repo_path(paths["cleaned_images_manifest"]), images)
    save_annotations(_resolve_repo_path(paths["cleaned_annotations_manifest"]), annotations)

    report = build_validation_report(images, annotations)
    write_json(_resolve_repo_path(paths["validation_report"]), report)

    if report["num_errors"]:
        raise SystemExit("dataset validation failed; inspect data/interim/cleaned/validation_report.json")


if __name__ == "__main__":
    main()
