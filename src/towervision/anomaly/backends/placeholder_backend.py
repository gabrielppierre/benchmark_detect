"""Placeholder proxy backend for the anomaly benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from towervision.anomaly.benchmark_dataset import read_dataset_manifest
from towervision.anomaly.benchmark_types import AnomalyBenchmarkDatasetRow, AnomalySeedRunResult
from towervision.anomaly.metrics import (
    classification_metrics_with_curves,
    select_threshold_for_f1,
    stratified_subset_metrics,
)
from towervision.utils.io import write_json


EPSILON = 1e-6


@dataclass(slots=True)
class ProxyModelArtifact:
    """Serializable anomaly model artifact."""

    model_name: str
    display_name: str
    backend: str
    fit_mode: str
    implementation_status: str
    training_rows: int
    feature_dim: int
    parameters: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "backend": self.backend,
            "fit_mode": self.fit_mode,
            "implementation_status": self.implementation_status,
            "training_rows": self.training_rows,
            "feature_dim": self.feature_dim,
            "parameters": self.parameters,
            "notes": self.notes,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Path to the anomaly benchmark job spec.")
    return parser.parse_args()


def _load_job(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid job payload: {path}")
    return payload


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _extract_feature_vector(crop_path: Path, *, input_size: int) -> np.ndarray:
    with Image.open(crop_path).convert("RGB") as image:
        resized = image.resize((input_size, input_size))
        array = np.asarray(resized, dtype=np.float32) / 255.0

    grayscale = array.mean(axis=2)
    gradients_x = np.diff(grayscale, axis=1, append=grayscale[:, -1:])
    gradients_y = np.diff(grayscale, axis=0, append=grayscale[-1:, :])
    edge_magnitude = np.sqrt(gradients_x * gradients_x + gradients_y * gradients_y)
    histogram = np.histogram(grayscale, bins=16, range=(0.0, 1.0), density=True)[0]

    half = input_size // 2
    quadrants = np.array(
        [
            grayscale[:half, :half].mean(),
            grayscale[:half, half:].mean(),
            grayscale[half:, :half].mean(),
            grayscale[half:, half:].mean(),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            array.mean(axis=(0, 1)),
            array.std(axis=(0, 1)),
            np.array(
                [
                    grayscale.mean(),
                    grayscale.std(),
                    edge_magnitude.mean(),
                    edge_magnitude.std(),
                ],
                dtype=np.float32,
            ),
            quadrants,
            histogram.astype(np.float32),
        ]
    )
    return features.astype(np.float32)


def _extract_rows_features(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    input_size: int,
) -> np.ndarray:
    return np.stack(
        [_extract_feature_vector(Path(row.crop_path), input_size=input_size) for row in rows],
        axis=0,
    )


def _fit_patchcore_proxy(train_features: np.ndarray) -> ProxyModelArtifact:
    memory_bank = train_features.astype(np.float32)
    return ProxyModelArtifact(
        model_name="patchcore",
        display_name="PatchCore",
        backend="placeholder",
        fit_mode="fit_once",
        implementation_status="proxy_nearest_neighbor",
        training_rows=int(len(train_features)),
        feature_dim=int(train_features.shape[1]),
        parameters={"memory_bank": memory_bank.tolist()},
        notes=["Proxy backend using handcrafted embeddings and nearest-neighbor distance."],
    )


def _score_patchcore_proxy(artifact: ProxyModelArtifact, features: np.ndarray) -> np.ndarray:
    memory_bank = np.asarray(artifact.parameters["memory_bank"], dtype=np.float32)
    pairwise = np.sqrt(((features[:, None, :] - memory_bank[None, :, :]) ** 2).sum(axis=2))
    return pairwise.min(axis=1)


def _fit_padim_proxy(train_features: np.ndarray) -> ProxyModelArtifact:
    mean_vector = train_features.mean(axis=0)
    std_vector = train_features.std(axis=0) + EPSILON
    return ProxyModelArtifact(
        model_name="padim",
        display_name="PaDiM",
        backend="placeholder",
        fit_mode="fit_once",
        implementation_status="proxy_diagonal_gaussian",
        training_rows=int(len(train_features)),
        feature_dim=int(train_features.shape[1]),
        parameters={
            "mean_vector": mean_vector.tolist(),
            "std_vector": std_vector.tolist(),
        },
        notes=["Proxy backend using handcrafted embeddings and diagonal Gaussian distance."],
    )


def _score_padim_proxy(artifact: ProxyModelArtifact, features: np.ndarray) -> np.ndarray:
    mean_vector = np.asarray(artifact.parameters["mean_vector"], dtype=np.float32)
    std_vector = np.asarray(artifact.parameters["std_vector"], dtype=np.float32)
    z_scores = (features - mean_vector[None, :]) / std_vector[None, :]
    return np.sqrt((z_scores * z_scores).mean(axis=1))


def _make_cutpaste_proxy_negatives(train_features: np.ndarray) -> np.ndarray:
    if len(train_features) <= 1:
        return train_features + 0.05
    rolled = np.roll(train_features, shift=1, axis=0)
    split_index = train_features.shape[1] // 2
    return np.concatenate(
        [train_features[:, :split_index], rolled[:, split_index:]],
        axis=1,
    )


def _fit_cutpaste_proxy(
    train_features: np.ndarray,
    *,
    val_labels: Sequence[int],
    val_features: np.ndarray,
    max_epochs: int,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-4,
    patience: int = 20,
    min_epochs: int = 25,
) -> tuple[ProxyModelArtifact, list[dict[str, float]], dict[str, float]]:
    proxy_negatives = _make_cutpaste_proxy_negatives(train_features)
    train_x = np.concatenate([train_features, proxy_negatives], axis=0)
    train_y = np.concatenate(
        [np.zeros(len(train_features), dtype=np.float32), np.ones(len(proxy_negatives), dtype=np.float32)],
        axis=0,
    )

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    bias = 0.0
    best_val_metrics: dict[str, float] | None = None
    best_payload: tuple[np.ndarray, float] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        logits = train_x @ weights + bias
        probabilities = _sigmoid(logits)
        loss = float(
            -np.mean(
                train_y * np.log(probabilities + EPSILON)
                + (1.0 - train_y) * np.log(1.0 - probabilities + EPSILON)
            )
        )
        gradient = probabilities - train_y
        weights -= learning_rate * ((train_x.T @ gradient) / len(train_x) + weight_decay * weights)
        bias -= learning_rate * float(gradient.mean())

        val_scores = _sigmoid(val_features @ weights + bias)
        val_threshold = select_threshold_for_f1(val_labels, val_scores.tolist(), recall_floor=0.90)
        val_metrics = classification_metrics_with_curves(
            val_labels,
            val_scores.tolist(),
            threshold=float(val_threshold["threshold"]),
        )
        epoch_time_seconds = time.perf_counter() - start
        improved = best_val_metrics is None or (
            val_metrics["roi_auroc"],
            val_metrics["roi_auprc"],
            val_metrics["f1"],
        ) > (
            best_val_metrics["roi_auroc"],
            best_val_metrics["roi_auprc"],
            best_val_metrics["f1"],
        )
        if improved:
            best_val_metrics = {"threshold": float(val_threshold["threshold"]), **val_metrics}
            best_payload = (weights.copy(), bias)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss_total": loss,
                "val_roi_auroc": float(val_metrics["roi_auroc"]),
                "val_roi_auprc": float(val_metrics["roi_auprc"]),
                "val_f1": float(val_metrics["f1"]),
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_time_seconds,
                "is_best": 1.0 if improved else 0.0,
            }
        )
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    if best_payload is None or best_val_metrics is None:
        best_payload = (weights, bias)
        val_scores = _sigmoid(val_features @ weights + bias)
        best_threshold = select_threshold_for_f1(val_labels, val_scores.tolist(), recall_floor=0.90)
        best_val_metrics = {
            "threshold": float(best_threshold["threshold"]),
            **classification_metrics_with_curves(
                val_labels,
                val_scores.tolist(),
                threshold=float(best_threshold["threshold"]),
            ),
        }
    best_weights, best_bias = best_payload
    artifact = ProxyModelArtifact(
        model_name="cutpaste",
        display_name="CutPaste",
        backend="placeholder",
        fit_mode="iterative",
        implementation_status="proxy_logistic_selfsupervised",
        training_rows=int(len(train_features)),
        feature_dim=int(train_features.shape[1]),
        parameters={"weights": best_weights.tolist(), "bias": float(best_bias)},
        notes=["Proxy backend using handcrafted embeddings and self-supervised logistic scoring."],
    )
    return artifact, history, best_val_metrics


def _score_cutpaste_proxy(artifact: ProxyModelArtifact, features: np.ndarray) -> np.ndarray:
    weights = np.asarray(artifact.parameters["weights"], dtype=np.float32)
    bias = float(artifact.parameters["bias"])
    return _sigmoid(features @ weights + bias)


def _score_rows(
    artifact: ProxyModelArtifact,
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    input_size: int,
) -> np.ndarray:
    features = _extract_rows_features(rows, input_size=input_size)
    if artifact.model_name == "patchcore":
        return _score_patchcore_proxy(artifact, features)
    if artifact.model_name == "padim":
        return _score_padim_proxy(artifact, features)
    return _score_cutpaste_proxy(artifact, features)


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss_total",
        "val_roi_auroc",
        "val_roi_auprc",
        "val_f1",
        "learning_rate",
        "epoch_time_seconds",
        "is_best",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _build_score_rows(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    scores: Sequence[float],
    threshold: float,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        payload.append(
            {
                "roi_id": row.roi_id,
                "image_id": row.image_id,
                "crop_path": row.crop_path,
                "split": row.split,
                "label": row.label,
                "score": float(score),
                "prediction": int(score >= threshold),
                "threshold": threshold,
                "source_kind": row.source_kind,
                "generator_family": row.generator_family,
                "anomaly_type": row.anomaly_type,
                "severity": row.severity,
                "mask_path": row.mask_path,
                "record_id": row.record_id,
                "pair_id": row.pair_id,
            }
        )
    return payload


def _write_scores_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "roi_id",
        "image_id",
        "crop_path",
        "split",
        "label",
        "score",
        "prediction",
        "threshold",
        "source_kind",
        "generator_family",
        "anomaly_type",
        "severity",
        "mask_path",
        "record_id",
        "pair_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_breakdown_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "group_field",
        "group_value",
        "normal_count",
        "anomaly_count",
        "total_count",
        "roi_auroc",
        "roi_auprc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_placeholder_job(job_path: Path) -> dict[str, Any]:
    """Execute one anomaly benchmark job with the proxy backend."""

    job = _load_job(job_path)
    run_dir = Path(str(job["run_dir"]))
    train_rows = read_dataset_manifest(Path(str(job["dataset_views"]["split_manifests"]["train"])))
    val_rows = read_dataset_manifest(Path(str(job["dataset_views"]["split_manifests"]["val"])))
    test_rows = read_dataset_manifest(Path(str(job["dataset_views"]["split_manifests"]["test"])))

    train_normals = [row for row in train_rows if row.label == 0]
    val_labels = [row.label for row in val_rows]
    test_labels = [row.label for row in test_rows]
    input_size = int(job["training"]["input_size"])
    train_features = _extract_rows_features(train_normals, input_size=input_size)
    val_features = _extract_rows_features(val_rows, input_size=input_size)

    model_name = str(job["model"]["name"])
    fit_mode = str(job["model"]["fit_mode"])
    if model_name == "patchcore":
        artifact = _fit_patchcore_proxy(train_features)
        val_scores = _score_patchcore_proxy(artifact, val_features)
        val_threshold_payload = select_threshold_for_f1(
            val_labels,
            val_scores.tolist(),
            recall_floor=float(job["ranking"]["operating_recall_floor"]),
        )
        train_history = [
            {
                "epoch": 1.0,
                "train_loss_total": 0.0,
                "val_roi_auroc": classification_metrics_with_curves(
                    val_labels,
                    val_scores.tolist(),
                    threshold=float(val_threshold_payload["threshold"]),
                )["roi_auroc"],
                "val_roi_auprc": classification_metrics_with_curves(
                    val_labels,
                    val_scores.tolist(),
                    threshold=float(val_threshold_payload["threshold"]),
                )["roi_auprc"],
                "val_f1": val_threshold_payload["f1"],
                "learning_rate": 0.0,
                "epoch_time_seconds": 0.0,
                "is_best": 1.0,
            }
        ]
    elif model_name == "padim":
        artifact = _fit_padim_proxy(train_features)
        val_scores = _score_padim_proxy(artifact, val_features)
        val_threshold_payload = select_threshold_for_f1(
            val_labels,
            val_scores.tolist(),
            recall_floor=float(job["ranking"]["operating_recall_floor"]),
        )
        train_history = [
            {
                "epoch": 1.0,
                "train_loss_total": 0.0,
                "val_roi_auroc": classification_metrics_with_curves(
                    val_labels,
                    val_scores.tolist(),
                    threshold=float(val_threshold_payload["threshold"]),
                )["roi_auroc"],
                "val_roi_auprc": classification_metrics_with_curves(
                    val_labels,
                    val_scores.tolist(),
                    threshold=float(val_threshold_payload["threshold"]),
                )["roi_auprc"],
                "val_f1": val_threshold_payload["f1"],
                "learning_rate": 0.0,
                "epoch_time_seconds": 0.0,
                "is_best": 1.0,
            }
        ]
    else:
        artifact, train_history, best_val_metrics = _fit_cutpaste_proxy(
            train_features,
            val_labels=val_labels,
            val_features=val_features,
            max_epochs=int(job["training"]["max_epochs"]),
            patience=int(job["training"]["patience"]),
            min_epochs=int(job["training"]["min_epochs"]),
        )
        val_scores = _score_cutpaste_proxy(artifact, val_features)
        val_threshold_payload = {
            "threshold": float(best_val_metrics["threshold"]),
            "f1": float(best_val_metrics["f1"]),
            "precision": float(best_val_metrics["precision"]),
            "recall": float(best_val_metrics["recall"]),
            "accuracy": float(best_val_metrics["accuracy"]),
            "tp": float(best_val_metrics["tp"]),
            "tn": float(best_val_metrics["tn"]),
            "fp": float(best_val_metrics["fp"]),
            "fn": float(best_val_metrics["fn"]),
        }

    threshold = float(val_threshold_payload["threshold"])
    val_metrics = {
        "threshold": threshold,
        **classification_metrics_with_curves(val_labels, val_scores.tolist(), threshold=threshold),
    }
    test_scores = _score_rows(artifact, test_rows, input_size=input_size)
    test_metrics = {
        "threshold": threshold,
        **classification_metrics_with_curves(test_labels, test_scores.tolist(), threshold=threshold),
    }

    val_score_rows = _build_score_rows(val_rows, scores=val_scores.tolist(), threshold=threshold)
    test_score_rows = _build_score_rows(test_rows, scores=test_scores.tolist(), threshold=threshold)
    val_normal_rows = [row for row in val_score_rows if int(row["label"]) == 0]
    val_anomaly_rows = [row for row in val_score_rows if int(row["label"]) == 1]
    test_normal_rows = [row for row in test_score_rows if int(row["label"]) == 0]
    test_anomaly_rows = [row for row in test_score_rows if int(row["label"]) == 1]
    generator_breakdown = stratified_subset_metrics(
        test_normal_rows,
        test_anomaly_rows,
        group_field="generator_family",
        threshold=threshold,
    )
    anomaly_type_breakdown = stratified_subset_metrics(
        test_normal_rows,
        test_anomaly_rows,
        group_field="anomaly_type",
        threshold=threshold,
    )
    severity_breakdown = stratified_subset_metrics(
        test_normal_rows,
        test_anomaly_rows,
        group_field="severity",
        threshold=threshold,
    )

    model_artifact_path = run_dir / "model.json"
    train_history_path = run_dir / "train_history.csv"
    threshold_selection_path = run_dir / "threshold_selection.json"
    val_scores_path = run_dir / "val_scores.csv"
    test_scores_path = run_dir / "test_scores.csv"
    val_metrics_path = run_dir / "val_metrics.json"
    test_metrics_path = run_dir / "test_metrics.json"
    generator_breakdown_path = run_dir / "generator_breakdown.csv"
    anomaly_type_breakdown_path = run_dir / "anomaly_type_breakdown.csv"
    severity_breakdown_path = run_dir / "severity_breakdown.csv"

    write_json(model_artifact_path, artifact.to_dict())
    _write_history_csv(train_history_path, train_history)
    write_json(
        threshold_selection_path,
        {
            "threshold": threshold,
            "selection_metric": "val_f1",
            "selection_recall_floor": float(job["ranking"]["operating_recall_floor"]),
            **val_threshold_payload,
        },
    )
    _write_scores_csv(val_scores_path, val_score_rows)
    _write_scores_csv(test_scores_path, test_score_rows)
    write_json(val_metrics_path, val_metrics)
    write_json(test_metrics_path, test_metrics)
    _write_breakdown_csv(generator_breakdown_path, generator_breakdown)
    _write_breakdown_csv(anomaly_type_breakdown_path, anomaly_type_breakdown)
    _write_breakdown_csv(severity_breakdown_path, severity_breakdown)

    result = AnomalySeedRunResult(
        model_name=model_name,
        display_name=str(job["model"]["display_name"]),
        seed=int(job["seed"]),
        status="completed",
        backend=str(job["model"]["backend"]),
        fit_mode=fit_mode,
        model_artifact_path=model_artifact_path.as_posix(),
        train_log_path=(run_dir / "train.log").as_posix(),
        train_history_path=train_history_path.as_posix(),
        threshold_selection_path=threshold_selection_path.as_posix(),
        val_scores_path=val_scores_path.as_posix(),
        test_scores_path=test_scores_path.as_posix(),
        val_metrics_path=val_metrics_path.as_posix(),
        test_metrics_path=test_metrics_path.as_posix(),
        generator_breakdown_path=generator_breakdown_path.as_posix(),
        anomaly_type_breakdown_path=anomaly_type_breakdown_path.as_posix(),
        severity_breakdown_path=severity_breakdown_path.as_posix(),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        notes=[
            f"implementation_status={artifact.implementation_status}",
            "Current backend is a repository-local proxy, not the reference external implementation.",
        ],
    ).to_dict()
    write_json(run_dir / "result.json", result)
    return result


def main() -> None:
    args = _parse_args()
    result = run_placeholder_job(args.job)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
