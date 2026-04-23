"""Common backend helpers for anomaly benchmark runners."""

from __future__ import annotations

import csv
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.pre_processing import PreProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize, Resize

from towervision.anomaly.benchmark_dataset import read_dataset_manifest
from towervision.anomaly.benchmark_types import AnomalyBenchmarkDatasetRow
from towervision.anomaly.metrics import (
    average_precision_score,
    classification_metrics_with_curves,
    roc_auc_score,
    select_threshold_for_f1,
    stratified_subset_metrics,
)
from towervision.utils.io import ensure_dir, write_json


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class FlattenedPrediction:
    """Prediction fields flattened from backend outputs."""

    image_path: str
    pred_score: float
    pred_label: int
    gt_label: int
    anomaly_map: np.ndarray | None = None
    gt_mask: np.ndarray | None = None


class RoiImageDataset(Dataset[dict[str, Any]]):
    """Simple ROI dataset for custom anomaly models."""

    def __init__(self, rows: Sequence[AnomalyBenchmarkDatasetRow], *, input_size: int) -> None:
        self.rows = list(rows)
        self.resize = Resize((input_size, input_size), antialias=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = _load_resized_image(Path(row.crop_path), input_size=None, resize=self.resize)
        sample: dict[str, Any] = {
            "image": image,
            "image_path": row.crop_path,
            "label": row.label,
            "roi_id": row.roi_id,
        }
        if row.mask_path:
            sample["mask"] = _load_resized_mask(Path(row.mask_path), input_size=None, resize=self.resize)
        else:
            sample["mask"] = None
        return sample


def load_job(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = write_back_compatible_json_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid job payload: {path}")
    return payload


def write_back_compatible_json_load(handle: Any) -> Any:
    import json

    return json.load(handle)


def build_anomalib_preprocessor(input_size: int) -> PreProcessor:
    """Build the common image resize and normalization pipeline."""

    return PreProcessor(
        transform=Compose(
            [
                Resize((input_size, input_size), antialias=True),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    )


def load_split_rows(job: Mapping[str, Any], split_name: str) -> list[AnomalyBenchmarkDatasetRow]:
    manifest_path = Path(str(job["dataset_views"]["split_manifests"][split_name]))
    return read_dataset_manifest(manifest_path)


def split_rows_by_label(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
) -> tuple[list[AnomalyBenchmarkDatasetRow], list[AnomalyBenchmarkDatasetRow]]:
    normal_rows = [row for row in rows if row.label == 0]
    anomaly_rows = [row for row in rows if row.label == 1]
    return normal_rows, anomaly_rows


def make_roi_dataloader(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    input_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = RoiImageDataset(rows, input_size=input_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def flatten_anomalib_predictions(prediction_batches: Sequence[Any]) -> list[FlattenedPrediction]:
    """Flatten anomalib ImageBatch predictions into per-image records."""

    flattened: list[FlattenedPrediction] = []
    for batch in prediction_batches:
        scores = getattr(batch, "pred_score", None)
        pred_labels = getattr(batch, "pred_label", None)
        gt_labels = getattr(batch, "gt_label", None)
        image_paths = list(getattr(batch, "image_path", []))
        anomaly_maps = getattr(batch, "anomaly_map", None)
        gt_masks = getattr(batch, "gt_mask", None)
        for index, image_path in enumerate(image_paths):
            score = _tensor_item(scores[index]) if scores is not None else 0.0
            pred_label = int(bool(_tensor_item(pred_labels[index]))) if pred_labels is not None else 0
            gt_label = int(bool(_tensor_item(gt_labels[index]))) if gt_labels is not None else 0
            anomaly_map = _tensor_numpy(anomaly_maps[index]) if anomaly_maps is not None else None
            gt_mask = _tensor_numpy(gt_masks[index]) if gt_masks is not None else None
            flattened.append(
                FlattenedPrediction(
                    image_path=str(image_path),
                    pred_score=_sanitize_score(score),
                    pred_label=pred_label,
                    gt_label=gt_label,
                    anomaly_map=anomaly_map,
                    gt_mask=gt_mask,
                )
            )
    return flattened


def build_score_rows(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    score_by_path: Mapping[str, FlattenedPrediction],
    threshold: float,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        prediction = _lookup_prediction(score_by_path, row.crop_path)
        payload.append(
            {
                "roi_id": row.roi_id,
                "image_id": row.image_id,
                "crop_path": row.crop_path,
                "split": row.split,
                "label": row.label,
                "score": prediction.pred_score,
                "prediction": int(prediction.pred_score >= threshold),
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


def compute_split_metrics(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    score_by_path: Mapping[str, FlattenedPrediction],
    threshold: float,
) -> dict[str, float]:
    labels = [row.label for row in rows]
    scores = [_lookup_prediction(score_by_path, row.crop_path).pred_score for row in rows]
    metrics = classification_metrics_with_curves(labels, scores, threshold=threshold)
    metrics["threshold"] = float(threshold)
    pixel_metrics = compute_pixel_metrics(rows, score_by_path=score_by_path)
    metrics.update(pixel_metrics)
    return metrics


def compute_pixel_metrics(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    score_by_path: Mapping[str, FlattenedPrediction],
) -> dict[str, float]:
    gt_pixels: list[int] = []
    score_pixels: list[float] = []
    for row in rows:
        prediction = _lookup_prediction(score_by_path, row.crop_path)
        if prediction.anomaly_map is None or prediction.gt_mask is None:
            continue
        anomaly_map = prediction.anomaly_map.astype(np.float32)
        gt_mask = prediction.gt_mask.astype(np.uint8)
        if anomaly_map.ndim != 2:
            continue
        gt_pixels.extend(gt_mask.reshape(-1).astype(int).tolist())
        score_pixels.extend(anomaly_map.reshape(-1).astype(float).tolist())
    if not gt_pixels:
        return {}
    return {
        "pixel_auroc": roc_auc_score(gt_pixels, score_pixels),
        "pixel_auprc": average_precision_score(gt_pixels, score_pixels),
    }


def compute_breakdowns(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    normal_rows = [row for row in score_rows if int(row["label"]) == 0]
    anomaly_rows = [row for row in score_rows if int(row["label"]) == 1]
    return {
        "generator_family": stratified_subset_metrics(
            normal_rows,
            anomaly_rows,
            group_field="generator_family",
            threshold=threshold,
        ),
        "anomaly_type": stratified_subset_metrics(
            normal_rows,
            anomaly_rows,
            group_field="anomaly_type",
            threshold=threshold,
        ),
        "severity": stratified_subset_metrics(
            normal_rows,
            anomaly_rows,
            group_field="severity",
            threshold=threshold,
        ),
    }


def write_scores_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_breakdown_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_train_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def persist_run_outputs(
    *,
    run_dir: Path,
    model_name: str,
    display_name: str,
    backend: str,
    fit_mode: str,
    seed: int,
    model_payload: Mapping[str, Any],
    train_history: Sequence[Mapping[str, Any]],
    threshold_payload: Mapping[str, Any],
    val_score_rows: Sequence[Mapping[str, Any]],
    test_score_rows: Sequence[Mapping[str, Any]],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    breakdowns: Mapping[str, Sequence[Mapping[str, Any]]],
    notes: Sequence[str],
) -> dict[str, Any]:
    """Write canonical run outputs and return the structured result payload."""

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

    write_json(model_artifact_path, dict(model_payload))
    write_train_history_csv(train_history_path, train_history)
    write_json(threshold_selection_path, dict(threshold_payload))
    write_scores_csv(val_scores_path, val_score_rows)
    write_scores_csv(test_scores_path, test_score_rows)
    write_json(val_metrics_path, dict(val_metrics))
    write_json(test_metrics_path, dict(test_metrics))
    write_breakdown_csv(generator_breakdown_path, breakdowns["generator_family"])
    write_breakdown_csv(anomaly_type_breakdown_path, breakdowns["anomaly_type"])
    write_breakdown_csv(severity_breakdown_path, breakdowns["severity"])

    result = {
        "model_name": model_name,
        "display_name": display_name,
        "seed": seed,
        "status": "completed",
        "backend": backend,
        "fit_mode": fit_mode,
        "model_artifact_path": model_artifact_path.as_posix(),
        "train_log_path": (run_dir / "train.log").as_posix(),
        "train_history_path": train_history_path.as_posix(),
        "threshold_selection_path": threshold_selection_path.as_posix(),
        "val_scores_path": val_scores_path.as_posix(),
        "test_scores_path": test_scores_path.as_posix(),
        "val_metrics_path": val_metrics_path.as_posix(),
        "test_metrics_path": test_metrics_path.as_posix(),
        "generator_breakdown_path": generator_breakdown_path.as_posix(),
        "anomaly_type_breakdown_path": anomaly_type_breakdown_path.as_posix(),
        "severity_breakdown_path": severity_breakdown_path.as_posix(),
        "val_metrics": dict(val_metrics),
        "test_metrics": dict(test_metrics),
        "notes": list(notes),
    }
    write_json(run_dir / "result.json", result)
    return result


def build_threshold_payload(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    recall_floor: float,
) -> dict[str, float]:
    threshold_payload = select_threshold_for_f1(labels, scores, recall_floor=recall_floor)
    return {
        "threshold": float(threshold_payload["threshold"]),
        "selection_metric": "val_f1",
        "selection_recall_floor": float(recall_floor),
        **threshold_payload,
    }


def flatten_rows_scores(
    rows: Sequence[AnomalyBenchmarkDatasetRow],
    *,
    score_by_path: Mapping[str, FlattenedPrediction],
) -> tuple[list[int], list[float]]:
    labels = [row.label for row in rows]
    scores = [_lookup_prediction(score_by_path, row.crop_path).pred_score for row in rows]
    return labels, scores


def _lookup_prediction(
    score_by_path: Mapping[str, FlattenedPrediction],
    crop_path: str,
) -> FlattenedPrediction:
    path = Path(crop_path)
    candidates = [crop_path, path.as_posix()]
    try:
        candidates.append(path.resolve().as_posix())
    except FileNotFoundError:
        pass
    for candidate in candidates:
        prediction = score_by_path.get(candidate)
        if prediction is not None:
            return prediction
    raise KeyError(crop_path)


def load_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def imagenet_normalize(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def _load_resized_image(path: Path, *, input_size: int | None, resize: Resize | None = None) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(path).convert("RGB") as image:
        if resize is not None:
            image = resize(image)
        elif input_size is not None:
            image = Resize((input_size, input_size), antialias=True)(image)
        tensor = pil_to_tensor(image).float() / 255.0
    return tensor


def _load_resized_mask(path: Path, *, input_size: int | None, resize: Resize | None = None) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(path).convert("L") as image:
        if resize is not None:
            image = resize(image)
        elif input_size is not None:
            image = Resize((input_size, input_size), antialias=True)(image)
        tensor = pil_to_tensor(image).float() / 255.0
    return (tensor > 0.5).float().squeeze(0)


def _tensor_item(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sanitize_score(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def timer() -> float:
    return time.perf_counter()
