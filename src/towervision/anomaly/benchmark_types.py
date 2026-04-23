"""Shared types for the anomaly benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ANOMALY_BENCHMARK_ROW_FIELDS = [
    "roi_id",
    "record_id",
    "pair_id",
    "image_id",
    "source_image_path",
    "source_crop_path",
    "crop_path",
    "mask_path",
    "split",
    "label",
    "source_kind",
    "generator_family",
    "anomaly_type",
    "severity",
]


@dataclass(slots=True)
class AnomalyBenchmarkTrainingConfig:
    """Shared training and calibration controls for anomaly methods."""

    train_only_normal: bool
    synthetic_in_training: bool
    input_size: int
    normalization: str
    feature_extractor: str
    num_seeds: int
    seeds: list[int]
    operating_recall_floor: float
    iterative_enabled_for: list[str] = field(default_factory=list)
    max_epochs: int = 100
    validate_every: int = 1
    save_best: bool = True
    save_last: bool = True
    early_stopping: bool = True
    monitor: str = "val_roi_auroc"
    mode: str = "max"
    patience: int = 20
    min_epochs: int = 25


@dataclass(slots=True)
class AnomalyBenchmarkModelConfig:
    """Per-model anomaly benchmark configuration."""

    name: str
    display_name: str
    family: str
    backend: str
    fit_mode: str
    input_size: int
    feature_extractor: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnomalyBenchmarkDatasetRow:
    """One ROI row consumed by the anomaly benchmark."""

    roi_id: str
    record_id: str
    pair_id: str
    image_id: str
    source_image_path: str
    source_crop_path: str
    crop_path: str
    mask_path: str
    split: str
    label: int
    source_kind: str
    generator_family: str = ""
    anomaly_type: str = ""
    severity: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AnomalyBenchmarkDatasetRow":
        """Build one dataset row from a raw mapping."""

        return cls(
            roi_id=str(raw.get("roi_id", "")),
            record_id=str(raw.get("record_id", "")),
            pair_id=str(raw.get("pair_id", "")),
            image_id=str(raw.get("image_id", "")),
            source_image_path=str(raw.get("source_image_path", "")),
            source_crop_path=str(raw.get("source_crop_path", "")),
            crop_path=str(raw.get("crop_path", "")),
            mask_path=str(raw.get("mask_path", "")),
            split=str(raw.get("split", "")),
            label=int(raw.get("label", 0)),
            source_kind=str(raw.get("source_kind", "")),
            generator_family=str(raw.get("generator_family", "")),
            anomaly_type=str(raw.get("anomaly_type", "")),
            severity=str(raw.get("severity", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the row to a serializable mapping."""

        return {
            "roi_id": self.roi_id,
            "record_id": self.record_id,
            "pair_id": self.pair_id,
            "image_id": self.image_id,
            "source_image_path": self.source_image_path,
            "source_crop_path": self.source_crop_path,
            "crop_path": self.crop_path,
            "mask_path": self.mask_path,
            "split": self.split,
            "label": self.label,
            "source_kind": self.source_kind,
            "generator_family": self.generator_family,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
        }


@dataclass(slots=True)
class AnomalyBenchmarkDatasetArtifacts:
    """Materialized views used by the anomaly benchmark."""

    root_dir: Path
    summary_path: Path
    split_to_root_dir: dict[str, Path]
    split_to_manifest_path: dict[str, Path]
    split_to_normal_dir: dict[str, Path]
    split_to_anomaly_dir: dict[str, Path]
    crop_padding: int


@dataclass(slots=True)
class AnomalySeedRunResult:
    """Structured final result for one anomaly model and seed."""

    model_name: str
    display_name: str
    seed: int
    status: str
    backend: str
    fit_mode: str
    model_artifact_path: str | None
    train_log_path: str | None
    train_history_path: str | None
    threshold_selection_path: str | None
    val_scores_path: str | None
    test_scores_path: str | None
    val_metrics_path: str | None
    test_metrics_path: str | None
    generator_breakdown_path: str | None
    anomaly_type_breakdown_path: str | None
    severity_breakdown_path: str | None
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the seed result to a JSON-serializable payload."""

        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "seed": self.seed,
            "status": self.status,
            "backend": self.backend,
            "fit_mode": self.fit_mode,
            "model_artifact_path": self.model_artifact_path,
            "train_log_path": self.train_log_path,
            "train_history_path": self.train_history_path,
            "threshold_selection_path": self.threshold_selection_path,
            "val_scores_path": self.val_scores_path,
            "test_scores_path": self.test_scores_path,
            "val_metrics_path": self.val_metrics_path,
            "test_metrics_path": self.test_metrics_path,
            "generator_breakdown_path": self.generator_breakdown_path,
            "anomaly_type_breakdown_path": self.anomaly_type_breakdown_path,
            "severity_breakdown_path": self.severity_breakdown_path,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "notes": self.notes,
        }
