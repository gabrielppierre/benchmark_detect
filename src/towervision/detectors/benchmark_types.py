"""Shared types for the fair detection benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CLASS_NAMES = ("torre", "isoladores")


@dataclass(slots=True)
class BenchmarkTrainingConfig:
    """Common training settings shared by all detector families."""

    img_size: int
    max_epochs: int
    validate_every: int
    save_best: bool
    save_last: bool
    early_stopping: bool
    monitor: str
    mode: str
    patience: int
    min_epochs: int
    num_seeds: int
    seeds: list[int]
    recall_floor_isoladores: float
    augmentations: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkModelConfig:
    """Per-model configuration used by the unified runner."""

    name: str
    display_name: str
    backend: str
    family: str
    pretrained_weights: str
    batch_size: int
    num_workers: int = 4
    confidence_threshold: float = 0.001
    nms_iou_threshold: float = 0.6
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkDatasetArtifacts:
    """Paths to the materialized dataset views used by the benchmark."""

    root_dir: Path
    coco_root: Path
    coco_annotations_dir: Path
    ultralytics_root: Path
    summary_path: Path
    split_to_image_dir: dict[str, Path]
    split_to_annotation_path: dict[str, Path]
    split_to_yolo_image_dir: dict[str, Path]
    split_to_yolo_label_dir: dict[str, Path]
    ultralytics_dataset_yaml: Path


@dataclass(slots=True)
class EpochMetrics:
    """Metrics recorded for one training epoch."""

    epoch: int
    train_loss_total: float | None = None
    component_losses: dict[str, float] = field(default_factory=dict)
    val_map50: float | None = None
    val_map50_95: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    class_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float | None = None
    epoch_time_seconds: float | None = None
    is_best: bool = False
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the epoch metrics to a JSON-serializable payload."""

        payload = {
            "epoch": self.epoch,
            "train_loss_total": self.train_loss_total,
            "val_map50": self.val_map50,
            "val_map50_95": self.val_map50_95,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "learning_rate": self.learning_rate,
            "epoch_time_seconds": self.epoch_time_seconds,
            "best_checkpoint": self.is_best,
            "checkpoint_path": self.checkpoint_path,
        }
        payload.update(self.component_losses)
        payload.update(self.class_metrics)
        return payload


@dataclass(slots=True)
class SeedRunResult:
    """Final structured result for one model and one seed."""

    model_name: str
    display_name: str
    seed: int
    status: str
    best_epoch: int | None
    best_checkpoint_path: str | None
    train_log_path: str | None
    epoch_metrics_path: str | None
    val_best_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the seed result to a JSON-serializable payload."""

        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "seed": self.seed,
            "status": self.status,
            "best_epoch": self.best_epoch,
            "best_checkpoint_path": self.best_checkpoint_path,
            "train_log_path": self.train_log_path,
            "epoch_metrics_path": self.epoch_metrics_path,
            "val_best_metrics": self.val_best_metrics,
            "test_metrics": self.test_metrics,
            "notes": self.notes,
        }
