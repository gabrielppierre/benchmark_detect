"""COCO evaluation helpers shared by detector backends."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco_detections(
    *,
    annotation_path: Path,
    detections: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, float]:
    """Evaluate COCO bbox detections and return the benchmark metric schema."""

    coco_gt = COCO(annotation_path.as_posix())
    if not detections:
        return _empty_metrics(class_names)

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "mAP50": float(evaluator.stats[1]),
        "mAP50_95": float(evaluator.stats[0]),
    }
    metrics.update(_per_class_ap(evaluator, class_names))
    metrics.update(
        _simple_precision_recall(
            coco_gt=coco_gt,
            detections=detections,
            class_names=class_names,
            iou_threshold=0.5,
            score_threshold=0.001,
        )
    )
    return metrics


def _per_class_ap(evaluator: COCOeval, class_names: list[str]) -> dict[str, float]:
    precision = evaluator.eval["precision"]
    iou_thresholds = evaluator.params.iouThrs
    category_ids = list(evaluator.params.catIds)
    area_index = 0
    max_det_index = len(evaluator.params.maxDets) - 1
    iou50_index = int(np.where(np.isclose(iou_thresholds, 0.5))[0][0])

    result: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        if class_index >= len(category_ids):
            result[f"AP50_{class_name}"] = 0.0
            result[f"AP50_95_{class_name}"] = 0.0
            continue

        class_precision = precision[:, :, class_index, area_index, max_det_index]
        class_precision = class_precision[class_precision > -1]
        class_precision_50 = precision[iou50_index, :, class_index, area_index, max_det_index]
        class_precision_50 = class_precision_50[class_precision_50 > -1]
        result[f"AP50_{class_name}"] = _safe_mean(class_precision_50)
        result[f"AP50_95_{class_name}"] = _safe_mean(class_precision)
    return result


def _simple_precision_recall(
    *,
    coco_gt: COCO,
    detections: list[dict[str, Any]],
    class_names: list[str],
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, float]:
    categories = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda category: category["id"])
    category_to_name = {int(category["id"]): str(category["name"]) for category in categories}
    detections_by_image_class: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    annotations_by_image_class: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for detection in detections:
        if float(detection.get("score", 0.0)) >= score_threshold:
            key = (int(detection["image_id"]), int(detection["category_id"]))
            detections_by_image_class[key].append(detection)

    for annotation in coco_gt.dataset.get("annotations", []):
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        key = (int(annotation["image_id"]), int(annotation["category_id"]))
        annotations_by_image_class[key].append(annotation)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    by_class: dict[str, tuple[int, int, int]] = {class_name: (0, 0, 0) for class_name in class_names}

    for category_id, class_name in category_to_name.items():
        tp = fp = fn = 0
        image_ids = set(coco_gt.getImgIds())
        for image_id in image_ids:
            ground_truth = annotations_by_image_class.get((image_id, category_id), [])
            predicted = sorted(
                detections_by_image_class.get((image_id, category_id), []),
                key=lambda item: float(item["score"]),
                reverse=True,
            )
            matched_gt: set[int] = set()
            for detection in predicted:
                match_index = _best_unmatched_match(
                    detection_bbox=detection["bbox"],
                    ground_truth=ground_truth,
                    matched_gt=matched_gt,
                    iou_threshold=iou_threshold,
                )
                if match_index is None:
                    fp += 1
                else:
                    tp += 1
                    matched_gt.add(match_index)
            fn += len(ground_truth) - len(matched_gt)

        by_class[class_name] = (tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    result = {
        "precision": _precision(total_tp, total_fp),
        "recall": _recall(total_tp, total_fn),
    }
    for class_name in class_names:
        tp, fp, fn = by_class[class_name]
        result[f"Precision_{class_name}"] = _precision(tp, fp)
        result[f"Recall_{class_name}"] = _recall(tp, fn)
    return result


def _best_unmatched_match(
    *,
    detection_bbox: list[float],
    ground_truth: list[dict[str, Any]],
    matched_gt: set[int],
    iou_threshold: float,
) -> int | None:
    best_index = None
    best_iou = iou_threshold
    for index, annotation in enumerate(ground_truth):
        if index in matched_gt:
            continue
        iou = _xywh_iou(detection_bbox, annotation["bbox"])
        if iou >= best_iou:
            best_index = index
            best_iou = iou
    return best_index


def _xywh_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def _empty_metrics(class_names: list[str]) -> dict[str, float]:
    metrics = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
    for class_name in class_names:
        metrics[f"AP50_{class_name}"] = 0.0
        metrics[f"AP50_95_{class_name}"] = 0.0
        metrics[f"Recall_{class_name}"] = 0.0
        metrics[f"Precision_{class_name}"] = 0.0
    return metrics


def _precision(tp: int, fp: int) -> float:
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def _recall(tp: int, fn: int) -> float:
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def _safe_mean(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.mean(values))
