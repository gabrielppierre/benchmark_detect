"""Run placeholder detector inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.load import load_images_manifest  # noqa: E402
from towervision.detectors.infer import infer_detector, save_predictions  # noqa: E402
from towervision.utils.io import read_json, read_yaml  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    params = read_yaml(ROOT / "params.yaml")
    paths = params["paths"]

    images = load_images_manifest(_resolve_repo_path(paths["cleaned_images_manifest"]))
    split_mapping = read_json(_resolve_repo_path(paths["splits_path"]), default={})
    model_artifact = read_json(_resolve_repo_path(paths["detector_run_dir"]) / "model.json", default={})

    test_ids = set(split_mapping.get("test", []))
    test_images = [image for image in images if image.id in test_ids]

    predictions = infer_detector(
        test_images,
        model_artifact=model_artifact,
        confidence_threshold=params["detector"]["confidence_threshold"],
    )
    save_predictions(_resolve_repo_path(paths["detector_predictions_path"]), predictions)


if __name__ == "__main__":
    main()
