"""Helpers to summarize the full Tower Vision pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from towervision.anomaly.infer import AnomalyScore
from towervision.anomaly.metrics import labeled_metrics_from_scores, summarize_scores
from towervision.data.load import AnnotationRecord
from towervision.detectors.metrics import evaluate_detections


def _crop_summary(manifest: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    unique_images = {str(item["image_id"]) for item in manifest}
    return {
        "count": float(len(manifest)),
        "unique_images": float(len(unique_images)),
    }


def build_pipeline_report(
    *,
    ground_truth: Sequence[AnnotationRecord],
    predictions: Sequence[AnnotationRecord],
    gt_crop_manifest: Sequence[Mapping[str, Any]],
    pred_crop_manifest: Sequence[Mapping[str, Any]],
    gt_scores: Sequence[AnomalyScore],
    pred_scores: Sequence[AnomalyScore],
    detection_iou_threshold: float = 0.5,
    anomaly_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate detection, crop and anomaly summaries."""

    return {
        "detection": evaluate_detections(
            ground_truth,
            predictions,
            iou_threshold=detection_iou_threshold,
        ),
        "crops": {
            "gt_crops": _crop_summary(gt_crop_manifest),
            "pred_crops": _crop_summary(pred_crop_manifest),
        },
        "anomaly": {
            "threshold": anomaly_threshold,
            "gt_crops": {
                "summary": summarize_scores(gt_scores),
                "classification": labeled_metrics_from_scores(
                    gt_scores,
                    threshold=anomaly_threshold,
                ),
            },
            "pred_crops": {
                "summary": summarize_scores(pred_scores),
                "classification": labeled_metrics_from_scores(
                    pred_scores,
                    threshold=anomaly_threshold,
                ),
            },
        },
    }


def render_benchmark_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact markdown report from aggregated metrics."""

    detection = report["detection"]
    gt_crops = report["crops"]["gt_crops"]
    pred_crops = report["crops"]["pred_crops"]
    anomaly = report["anomaly"]

    lines = [
        "# Benchmark V1",
        "",
        "## Detection",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision | {detection['precision']:.4f} |",
        f"| Recall | {detection['recall']:.4f} |",
        f"| F1 | {detection['f1']:.4f} |",
        "",
        "## ROI Crops",
        "",
        "| Source | Count | Unique images |",
        "| --- | ---: | ---: |",
        f"| gt_crops | {gt_crops['count']:.0f} | {gt_crops['unique_images']:.0f} |",
        f"| pred_crops | {pred_crops['count']:.0f} | {pred_crops['unique_images']:.0f} |",
        "",
        "## Anomaly Summary",
        "",
        "| Source | Mean score | Max score |",
        "| --- | ---: | ---: |",
        (
            f"| gt_crops | {anomaly['gt_crops']['summary']['mean_score']:.4f} | "
            f"{anomaly['gt_crops']['summary']['max_score']:.4f} |"
        ),
        (
            f"| pred_crops | {anomaly['pred_crops']['summary']['mean_score']:.4f} | "
            f"{anomaly['pred_crops']['summary']['max_score']:.4f} |"
        ),
        "",
        "## Notes",
        "",
        "- `gt_crops` and `pred_crops` are tracked separately by design.",
        "- Placeholder adapters should be replaced by real model integrations over time.",
    ]
    return "\n".join(lines)
