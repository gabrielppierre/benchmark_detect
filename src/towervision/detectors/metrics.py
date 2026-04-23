"""Basic metrics for object detection."""

from __future__ import annotations

from collections.abc import Sequence

from towervision.data.load import AnnotationRecord


def bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two bounding boxes."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = (aw * ah) + (bw * bh) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall and F1 from counts."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_detections(
    ground_truth: Sequence[AnnotationRecord],
    predictions: Sequence[AnnotationRecord],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Greedy one-to-one matching between predictions and ground truth."""

    matched_gt: set[str] = set()
    tp = 0
    fp = 0

    ordered_predictions = sorted(
        predictions,
        key=lambda prediction: prediction.score if prediction.score is not None else 0.0,
        reverse=True,
    )

    for prediction in ordered_predictions:
        best_match_id: str | None = None
        best_iou = 0.0

        for annotation in ground_truth:
            if annotation.id in matched_gt:
                continue
            if annotation.image_id != prediction.image_id or annotation.label != prediction.label:
                continue
            iou = bbox_iou(annotation.bbox, prediction.bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_match_id = annotation.id

        if best_match_id is None:
            fp += 1
            continue

        matched_gt.add(best_match_id)
        tp += 1

    fn = len(ground_truth) - len(matched_gt)
    return precision_recall_f1(tp, fp, fn)
