from towervision.anomaly.metrics import (
    average_precision_score,
    binary_classification_metrics,
    roc_auc_score,
    select_threshold_for_f1,
)
from towervision.data.load import AnnotationRecord
from towervision.detectors.metrics import evaluate_detections


def test_detection_metrics_basic_counts() -> None:
    ground_truth = [
        AnnotationRecord(id="gt-1", image_id="img-1", bbox=(0, 0, 10, 10)),
    ]
    predictions = [
        AnnotationRecord(
            id="pred-1",
            image_id="img-1",
            bbox=(0, 0, 10, 10),
            score=0.9,
            source="pred",
        ),
        AnnotationRecord(
            id="pred-2",
            image_id="img-1",
            bbox=(20, 20, 5, 5),
            score=0.6,
            source="pred",
        ),
    ]

    metrics = evaluate_detections(ground_truth, predictions, iou_threshold=0.5)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 0.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_binary_anomaly_metrics() -> None:
    metrics = binary_classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.7, 0.6], threshold=0.5)

    assert metrics["tp"] == 2.0
    assert metrics["tn"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 0.0
    assert metrics["accuracy"] == 0.75


def test_threshold_free_anomaly_metrics() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert roc_auc_score(labels, scores) == 1.0
    assert average_precision_score(labels, scores) == 1.0


def test_select_threshold_for_f1_respects_recall_floor() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.2, 0.4, 0.6, 0.7]

    selected = select_threshold_for_f1(labels, scores, recall_floor=1.0)

    assert selected["recall"] == 1.0
    assert selected["threshold"] <= 0.6
