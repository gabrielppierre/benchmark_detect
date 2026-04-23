"""Metrics used by the anomaly benchmark."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from towervision.anomaly.infer import AnomalyScore


def _validate_binary_inputs(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")


def binary_classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary metrics from labels and scores."""

    _validate_binary_inputs(labels, scores)

    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(int(prediction == 1 and label == 1) for prediction, label in zip(predictions, labels))
    tn = sum(int(prediction == 0 and label == 0) for prediction, label in zip(predictions, labels))
    fp = sum(int(prediction == 1 and label == 0) for prediction, label in zip(predictions, labels))
    fn = sum(int(prediction == 0 and label == 1) for prediction, label in zip(predictions, labels))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average_rank
        cursor = end
    return ranks


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC AUC without external dependencies."""

    _validate_binary_inputs(labels, scores)
    positive_count = sum(int(label == 1) for label in labels)
    negative_count = sum(int(label == 0) for label in labels)
    if positive_count == 0 or negative_count == 0:
        return 0.0

    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def average_precision_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute step-wise average precision, used here as AUPRC."""

    _validate_binary_inputs(labels, scores)
    positive_count = sum(int(label == 1) for label in labels)
    if positive_count == 0:
        return 0.0

    ordered = sorted(
        zip(scores, labels),
        key=lambda item: item[0],
        reverse=True,
    )
    tp = 0
    fp = 0
    precision_sum = 0.0
    for _, label in ordered:
        if label == 1:
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1
    return float(precision_sum / positive_count)


def threshold_candidates(scores: Sequence[float]) -> list[float]:
    """Build sorted threshold candidates from score values."""

    candidates = {0.0, 1.0}
    candidates.update(float(score) for score in scores)
    return sorted(candidates)


def select_threshold_for_f1(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    recall_floor: float = 0.0,
) -> dict[str, float]:
    """Choose the operational threshold in validation."""

    _validate_binary_inputs(labels, scores)
    best_payload: dict[str, float] | None = None
    fallback_payload: dict[str, float] | None = None

    for threshold in threshold_candidates(scores):
        metrics = binary_classification_metrics(labels, scores, threshold=threshold)
        payload = {"threshold": float(threshold), **metrics}

        if fallback_payload is None or (
            payload["f1"],
            payload["precision"],
            -payload["threshold"],
        ) > (
            fallback_payload["f1"],
            fallback_payload["precision"],
            -fallback_payload["threshold"],
        ):
            fallback_payload = payload

        if payload["recall"] < recall_floor:
            continue
        if best_payload is None or (
            payload["f1"],
            payload["precision"],
            payload["accuracy"],
            -payload["threshold"],
        ) > (
            best_payload["f1"],
            best_payload["precision"],
            best_payload["accuracy"],
            -best_payload["threshold"],
        ):
            best_payload = payload

    return best_payload or fallback_payload or {
        "threshold": 0.5,
        **binary_classification_metrics(labels, scores, threshold=0.5),
    }


def classification_metrics_with_curves(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    """Compute the full ROI-level metrics bundle used by the benchmark."""

    return {
        "roi_auroc": roc_auc_score(labels, scores),
        "roi_auprc": average_precision_score(labels, scores),
        **binary_classification_metrics(labels, scores, threshold=threshold),
    }


def stratified_subset_metrics(
    normal_rows: Sequence[Mapping[str, Any]],
    anomaly_rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    threshold: float,
) -> list[dict[str, float | str]]:
    """Compare each anomaly subgroup against the same normal reference set."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in anomaly_rows:
        group_value = str(row.get(group_field, "") or "unknown")
        grouped[group_value].append(row)

    breakdown: list[dict[str, float | str]] = []
    normal_labels = [int(row["label"]) for row in normal_rows]
    normal_scores = [float(row["score"]) for row in normal_rows]
    for group_value, group_rows in sorted(grouped.items()):
        labels = normal_labels + [int(row["label"]) for row in group_rows]
        scores = normal_scores + [float(row["score"]) for row in group_rows]
        metrics = classification_metrics_with_curves(
            labels,
            scores,
            threshold=threshold,
        )
        breakdown.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "normal_count": float(len(normal_rows)),
                "anomaly_count": float(len(group_rows)),
                "total_count": float(len(labels)),
                **metrics,
            }
        )
    return breakdown


def labeled_metrics_from_scores(
    scores: Sequence[AnomalyScore],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute metrics only for crops that carry labels."""

    labeled_scores = [score for score in scores if score.label is not None]
    if not labeled_scores:
        return {
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    labels = [int(score.label) for score in labeled_scores]
    numeric_scores = [score.score for score in labeled_scores]
    return binary_classification_metrics(labels, numeric_scores, threshold=threshold)


def summarize_scores(scores: Sequence[AnomalyScore]) -> dict[str, float]:
    """Summarize anomaly scores without requiring labels."""

    if not scores:
        return {"count": 0.0, "mean_score": 0.0, "max_score": 0.0, "min_score": 0.0}

    numeric_scores = [score.score for score in scores]
    return {
        "count": float(len(scores)),
        "mean_score": sum(numeric_scores) / len(numeric_scores),
        "max_score": max(numeric_scores),
        "min_score": min(numeric_scores),
    }
