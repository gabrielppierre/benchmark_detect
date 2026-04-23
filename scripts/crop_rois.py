"""Generate ROI crops from ground truth and/or predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.load import load_annotations, load_images_manifest, index_images_by_id  # noqa: E402
from towervision.pipelines.crop_from_gt import crop_from_ground_truth  # noqa: E402
from towervision.pipelines.crop_from_pred import crop_from_predictions  # noqa: E402
from towervision.utils.io import clean_directory, read_yaml, write_json  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("gt", "pred", "both"),
        default="both",
        help="Seleciona quais crops gerar.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = read_yaml(ROOT / "params.yaml")
    paths = params["paths"]

    images = load_images_manifest(_resolve_repo_path(paths["cleaned_images_manifest"]))
    images_by_id = index_images_by_id(images)
    annotations = load_annotations(
        _resolve_repo_path(paths["cleaned_annotations_manifest"]),
        default_source="gt",
        allow_missing=True,
    )
    predictions = load_annotations(
        _resolve_repo_path(paths["detector_predictions_path"]),
        default_source="pred",
        allow_missing=True,
    )

    if args.source in {"gt", "both"}:
        gt_dir = clean_directory(_resolve_repo_path(paths["gt_crops_dir"]))
        gt_manifest = crop_from_ground_truth(
            images_by_id,
            annotations,
            output_dir=gt_dir,
            padding=params["crop"]["padding"],
        )
        write_json(gt_dir / "manifest.json", gt_manifest)

    if args.source in {"pred", "both"}:
        pred_dir = clean_directory(_resolve_repo_path(paths["pred_crops_dir"]))
        pred_manifest = crop_from_predictions(
            images_by_id,
            predictions,
            output_dir=pred_dir,
            score_threshold=params["detector"]["confidence_threshold"],
            padding=params["crop"]["padding"],
        )
        write_json(pred_dir / "manifest.json", pred_manifest)


if __name__ == "__main__":
    main()
