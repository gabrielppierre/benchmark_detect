"""Evaluate the full placeholder pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.anomaly.infer import infer_anomaly_scores  # noqa: E402
from towervision.data.load import load_annotations  # noqa: E402
from towervision.pipelines.end_to_end import build_pipeline_report, render_benchmark_markdown  # noqa: E402
from towervision.utils.io import read_json, read_yaml, write_json, write_text  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    params = read_yaml(ROOT / "params.yaml")
    paths = params["paths"]

    ground_truth = load_annotations(
        _resolve_repo_path(paths["cleaned_annotations_manifest"]),
        default_source="gt",
        allow_missing=True,
    )
    predictions = load_annotations(
        _resolve_repo_path(paths["detector_predictions_path"]),
        default_source="pred",
        allow_missing=True,
    )

    gt_dir = _resolve_repo_path(paths["gt_crops_dir"])
    pred_dir = _resolve_repo_path(paths["pred_crops_dir"])
    gt_crop_manifest = read_json(gt_dir / "manifest.json", default=[])
    pred_crop_manifest = read_json(pred_dir / "manifest.json", default=[])

    gt_scores = infer_anomaly_scores(gt_dir, source="gt_crops")
    pred_scores = infer_anomaly_scores(pred_dir, source="pred_crops")

    report = build_pipeline_report(
        ground_truth=ground_truth,
        predictions=predictions,
        gt_crop_manifest=gt_crop_manifest,
        pred_crop_manifest=pred_crop_manifest,
        gt_scores=gt_scores,
        pred_scores=pred_scores,
        detection_iou_threshold=params["evaluation"]["detection_iou_threshold"],
        anomaly_threshold=params["anomaly"]["threshold"],
    )

    write_json(_resolve_repo_path(paths["evaluation_report"]), report)
    write_text(_resolve_repo_path(paths["benchmark_report"]), render_benchmark_markdown(report))


if __name__ == "__main__":
    main()
