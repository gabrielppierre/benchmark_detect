"""Ultralytics backend for the fair detection benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from towervision.utils.io import read_json, write_json

PRIMARY_VAL_KEY = "metrics/mAP50-95(B)"


class MinEpochEarlyStopping:
    """Wrap Ultralytics early stopping so it cannot stop before min_epochs."""

    def __init__(self, inner: Any, min_epochs: int) -> None:
        self.inner = inner
        self.min_epochs = min_epochs
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float | None) -> bool:
        should_stop = bool(self.inner(epoch, fitness))
        self.possible_stop = bool(getattr(self.inner, "possible_stop", False) and epoch >= self.min_epochs)
        return should_stop and epoch >= self.min_epochs

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class FairDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer with fitness pinned to val mAP50-95."""

    min_epochs: int = 1

    def _setup_train(self) -> None:  # type: ignore[override]
        """Set up training and enforce a minimum epoch count before stopping."""

        super()._setup_train()
        if self.min_epochs > 1:
            self.stopper = MinEpochEarlyStopping(self.stopper, self.min_epochs)

    def validate(self):  # type: ignore[override]
        """Run validation and expose per-class metrics in the saved results CSV."""

        metrics = self.validator(self)
        if metrics is None:
            return None, None

        det_metrics = getattr(self.validator, "metrics", None)
        names = getattr(det_metrics, "names", {}) or {}
        if det_metrics is not None:
            for class_index, class_name in names.items():
                precision, recall, ap50, ap50_95 = det_metrics.class_result(class_index)
                metrics[f"AP50_{class_name}"] = float(ap50)
                metrics[f"AP50_95_{class_name}"] = float(ap50_95)
                metrics[f"Recall_{class_name}"] = float(recall)
                metrics[f"Precision_{class_name}"] = float(precision)

        fitness = float(metrics.get(PRIMARY_VAL_KEY, metrics.get("fitness", 0.0)))
        metrics["fitness"] = fitness
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Path to the benchmark job spec.")
    return parser.parse_args()


def load_job(path: Path) -> dict[str, Any]:
    """Load the JSON job spec."""

    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        raise ValueError(f"invalid job spec: {path}")
    return payload


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one Ultralytics benchmark job."""

    run_dir = Path(job["run_dir"])
    model_cfg = job["model"]
    training = job["training"]
    dataset_views = job["dataset_views"]

    pretrained_weights = model_cfg["pretrained_weights"]
    model = YOLO(pretrained_weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    FairDetectionTrainer.min_epochs = int(training["min_epochs"])
    model.train(
        data=dataset_views["ultralytics_dataset_yaml"],
        imgsz=training["img_size"],
        epochs=training["max_epochs"],
        batch=model_cfg["batch_size"],
        workers=model_cfg["num_workers"],
        device=device,
        seed=training["seed"],
        patience=training["patience"] if training["early_stopping"] else 0,
        save=training["save_best"] or training["save_last"],
        save_period=-1,
        val=True,
        verbose=True,
        pretrained=True,
        exist_ok=True,
        project=run_dir.parent.as_posix(),
        name=run_dir.name,
        trainer=FairDetectionTrainer,
        fliplr=float(training["augmentations"].get("horizontal_flip", 0.5)),
        scale=float(training["augmentations"].get("scale_jitter", [1.0, 1.0])[1]) - 1.0,
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
    )

    weights_dir = run_dir / "weights"
    best_checkpoint = weights_dir / "best.pt"
    last_checkpoint = weights_dir / "last.pt"
    epoch_metrics_path = convert_results_csv(run_dir / "results.csv", run_dir / "epoch_metrics.csv")
    best_epoch = infer_best_epoch(epoch_metrics_path)

    val_metrics = evaluate_checkpoint(
        checkpoint_path=best_checkpoint,
        dataset_yaml=Path(dataset_views["ultralytics_dataset_yaml"]),
        split="val",
        imgsz=training["img_size"],
        batch_size=model_cfg["batch_size"],
        num_workers=model_cfg["num_workers"],
        output_dir=run_dir / "val_eval",
        device=device,
    )
    test_metrics = evaluate_checkpoint(
        checkpoint_path=best_checkpoint,
        dataset_yaml=Path(dataset_views["ultralytics_dataset_yaml"]),
        split="test",
        imgsz=training["img_size"],
        batch_size=model_cfg["batch_size"],
        num_workers=model_cfg["num_workers"],
        output_dir=run_dir / "test_eval",
        device=device,
    )

    result = {
        "model_name": model_cfg["name"],
        "display_name": model_cfg["display_name"],
        "seed": training["seed"],
        "status": "completed",
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint.as_posix() if best_checkpoint.exists() else None,
        "last_checkpoint_path": last_checkpoint.as_posix() if last_checkpoint.exists() else None,
        "train_log_path": (run_dir / "train.log").as_posix(),
        "epoch_metrics_path": epoch_metrics_path.as_posix(),
        "val_best_metrics": val_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "ultralytics backend executed with custom trainer fitness tied to val_map50_95",
        ],
    }
    write_json(run_dir / "result.json", result)
    return result


def convert_results_csv(source_path: Path, output_path: Path) -> Path:
    """Normalize the Ultralytics results CSV into the benchmark epoch schema."""

    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            component_losses = {
                key.replace("/", "_"): float(value)
                for key, value in row.items()
                if key.startswith("train/") and key.endswith("loss") and value not in {"", None}
            }
            learning_rates = [
                float(value)
                for key, value in row.items()
                if key.startswith("lr/") and value not in {"", None}
            ]
            rows.append(
                {
                    "epoch": int(float(row["epoch"])),
                    "train_loss_total": sum(component_losses.values()) if component_losses else None,
                    **component_losses,
                    "val_map50": _safe_float(row.get("metrics/mAP50(B)")),
                    "val_map50_95": _safe_float(row.get(PRIMARY_VAL_KEY)),
                    "val_precision": _safe_float(row.get("metrics/precision(B)")),
                    "val_recall": _safe_float(row.get("metrics/recall(B)")),
                    "AP50_torre": _safe_float(row.get("AP50_torre")),
                    "AP50_95_torre": _safe_float(row.get("AP50_95_torre")),
                    "Recall_torre": _safe_float(row.get("Recall_torre")),
                    "Precision_torre": _safe_float(row.get("Precision_torre")),
                    "AP50_isoladores": _safe_float(row.get("AP50_isoladores")),
                    "AP50_95_isoladores": _safe_float(row.get("AP50_95_isoladores")),
                    "Recall_isoladores": _safe_float(row.get("Recall_isoladores")),
                    "Precision_isoladores": _safe_float(row.get("Precision_isoladores")),
                    "learning_rate": max(learning_rates) if learning_rates else None,
                    "epoch_time_seconds": _safe_float(row.get("time")),
                    "best_checkpoint": False,
                    "checkpoint_path": None,
                }
            )

    best_epoch = infer_best_epoch_from_rows(rows)
    previous_time: float | None = None
    for row in rows:
        cumulative_time = _safe_float(row.get("epoch_time_seconds"))
        row["epoch_time_seconds"] = (
            None if cumulative_time is None else cumulative_time - (previous_time or 0.0)
        )
        previous_time = cumulative_time
        if row["epoch"] == best_epoch:
            row["best_checkpoint"] = True
            row["checkpoint_path"] = (output_path.parent / "weights" / "best.pt").as_posix()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "epoch",
            "train_loss_total",
            "train_box_loss",
            "train_cls_loss",
            "train_dfl_loss",
            "val_map50",
            "val_map50_95",
            "val_precision",
            "val_recall",
            "AP50_torre",
            "AP50_95_torre",
            "Recall_torre",
            "Precision_torre",
            "AP50_isoladores",
            "AP50_95_isoladores",
            "Recall_isoladores",
            "Precision_isoladores",
            "learning_rate",
            "epoch_time_seconds",
            "best_checkpoint",
            "checkpoint_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return output_path


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    dataset_yaml: Path,
    split: str,
    imgsz: int,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    device: int | str,
) -> dict[str, float]:
    """Evaluate one checkpoint and extract harmonized metrics."""

    model = YOLO(checkpoint_path)
    metrics = model.val(
        data=dataset_yaml.as_posix(),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        workers=num_workers,
        device=device,
        conf=0.001,
        iou=0.7,
        save_json=True,
        verbose=True,
        project=output_dir.parent.as_posix(),
        name=output_dir.name,
    )
    box = metrics.box
    names = metrics.names
    results = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    for class_index, class_name in names.items():
        precision, recall, ap50, ap50_95 = box.class_result(class_index)
        results[f"AP50_{class_name}"] = float(ap50)
        results[f"AP50_95_{class_name}"] = float(ap50_95)
        results[f"Recall_{class_name}"] = float(recall)
        results[f"Precision_{class_name}"] = float(precision)
    return results


def infer_best_epoch(path: Path) -> int | None:
    """Infer the best epoch from the standardized epoch metrics CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    return infer_best_epoch_from_rows(rows)


def infer_best_epoch_from_rows(rows: list[dict[str, Any]]) -> int | None:
    """Find the epoch with highest val_map50_95."""

    best_row = None
    for row in rows:
        metric = _safe_float(row.get("val_map50_95"))
        epoch = int(float(row["epoch"]))
        if metric is None:
            continue
        if best_row is None or (metric, -epoch) > (best_row[0], -best_row[1]):
            best_row = (metric, epoch)
    return None if best_row is None else best_row[1]


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def main() -> None:
    """CLI entrypoint for one Ultralytics benchmark job."""

    args = parse_args()
    result = run_job(load_job(args.job))
    print(Path(args.job).as_posix())
    print(result["status"])
    print(result["best_checkpoint_path"])


if __name__ == "__main__":
    main()
