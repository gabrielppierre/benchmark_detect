"""Transformers RT-DETR backend for the fair detection benchmark."""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, RTDetrForObjectDetection

from towervision.detectors.backends.coco_eval import evaluate_coco_detections
from towervision.utils.io import read_json, write_json


class RTDetrCocoDataset(Dataset[tuple[Image.Image, dict[str, Any], tuple[int, int]]]):
    """Minimal COCO dataset wrapper for RT-DETR fine-tuning."""

    def __init__(
        self,
        *,
        image_dir: Path,
        annotation_path: Path,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        payload = read_json(annotation_path, default={})
        self.image_dir = image_dir
        self.images = list(payload.get("images", []))
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.horizontal_flip_prob = horizontal_flip_prob
        for annotation in payload.get("annotations", []):
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any], tuple[int, int]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.image_dir / str(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        annotations = [
            {
                "bbox": [
                    float(annotation["bbox"][0]),
                    float(annotation["bbox"][1]),
                    float(annotation["bbox"][2]),
                    float(annotation["bbox"][3]),
                ],
                "category_id": int(annotation["category_id"]) - 1,
                "area": float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
            for annotation in self.annotations_by_image.get(image_id, [])
        ]

        if self.horizontal_flip_prob > 0.0 and random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for annotation in annotations:
                x, y, box_width, box_height = annotation["bbox"]
                annotation["bbox"] = [width - x - box_width, y, box_width, box_height]

        target = {"image_id": image_id, "annotations": annotations}
        return image, target, (height, width)


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
    """Execute one RT-DETR benchmark job."""

    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = job["model"]
    training = job["training"]
    dataset_views = job["dataset_views"]
    class_names = list(job["class_names"])
    seed = int(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = str(model_cfg["pretrained_weights"])

    seed_everything(seed)
    id2label = {index: label for index, label in enumerate(class_names)}
    label2id = {label: index for index, label in id2label.items()}
    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        size={"height": int(training["img_size"]), "width": int(training["img_size"])},
    )
    model = RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_loader = build_loader(
        image_processor=image_processor,
        job=job,
        split="train",
        train=True,
    )
    val_loader = build_loader(
        image_processor=image_processor,
        job=job,
        split="val",
        train=False,
    )
    test_loader = build_loader(
        image_processor=image_processor,
        job=job,
        split="test",
        train=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("extra", {}).get("lr", 1e-4)),
        weight_decay=float(model_cfg.get("extra", {}).get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(training["max_epochs"]), 1),
    )

    best_metric = float("-inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    best_checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(training["max_epochs"]) + 1):
        started_at = time.perf_counter()
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()
        val_metrics, _ = evaluate_model(
            model=model,
            image_processor=image_processor,
            loader=val_loader,
            annotation_path=Path(dataset_views["coco_annotations"]["val"]),
            class_names=class_names,
            output_path=run_dir / "val_predictions_latest.json",
            device=device,
        )
        val_map = float(val_metrics["mAP50_95"])
        improved = val_map > best_metric
        if improved:
            best_metric = val_map
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                job=job,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            job=job,
        )
        epoch_rows.append(
            build_epoch_row(
                epoch=epoch,
                train_losses=train_losses,
                val_metrics=val_metrics,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                epoch_time_seconds=time.perf_counter() - started_at,
                best_checkpoint=improved,
                checkpoint_path=best_checkpoint.as_posix() if improved else None,
            )
        )
        print(
            f"epoch={epoch} val_map50_95={val_map:.4f} "
            f"best={best_metric:.4f} no_improve={epochs_without_improvement}"
        )

        if should_stop_early(
            epoch=epoch,
            epochs_without_improvement=epochs_without_improvement,
            training=training,
        ):
            print(f"early_stopping=true epoch={epoch} best_epoch={best_epoch}")
            break

    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    write_epoch_metrics(epoch_metrics_path, epoch_rows)
    best_model = load_model_for_eval(
        model_name=model_name,
        checkpoint_path=best_checkpoint,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        device=device,
    )
    best_val_metrics, val_predictions = evaluate_model(
        model=best_model,
        image_processor=image_processor,
        loader=val_loader,
        annotation_path=Path(dataset_views["coco_annotations"]["val"]),
        class_names=class_names,
        output_path=run_dir / "val_eval" / "predictions.json",
        device=device,
    )
    test_metrics, test_predictions = evaluate_model(
        model=best_model,
        image_processor=image_processor,
        loader=test_loader,
        annotation_path=Path(dataset_views["coco_annotations"]["test"]),
        class_names=class_names,
        output_path=run_dir / "test_eval" / "predictions.json",
        device=device,
    )

    result = {
        "model_name": model_cfg["name"],
        "display_name": model_cfg["display_name"],
        "seed": seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint.as_posix() if best_checkpoint.exists() else None,
        "last_checkpoint_path": last_checkpoint.as_posix() if last_checkpoint.exists() else None,
        "train_log_path": (run_dir / "train.log").as_posix(),
        "epoch_metrics_path": epoch_metrics_path.as_posix(),
        "val_best_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "transformers backend executed with RT-DETR-R18 and early stopping by val_map50_95",
            "precision/recall are simple IoU50 matching metrics at score >= 0.001",
        ],
    }
    write_json(run_dir / "result.json", result)
    return result


def build_loader(
    *,
    image_processor: Any,
    job: dict[str, Any],
    split: str,
    train: bool,
) -> DataLoader:
    """Build a deterministic data loader for one split."""

    model_cfg = job["model"]
    training = job["training"]
    dataset_views = job["dataset_views"]
    dataset = RTDetrCocoDataset(
        image_dir=Path(dataset_views["coco_images"][split]),
        annotation_path=Path(dataset_views["coco_annotations"][split]),
        horizontal_flip_prob=float(training["augmentations"].get("horizontal_flip", 0.0)) if train else 0.0,
    )
    generator = torch.Generator()
    generator.manual_seed(int(training["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(model_cfg["num_workers"]),
        collate_fn=lambda batch: collate_rtdetr_batch(batch, image_processor=image_processor),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def collate_rtdetr_batch(batch: list[tuple[Image.Image, dict[str, Any], tuple[int, int]]], *, image_processor: Any) -> dict[str, Any]:
    """Encode a detection batch with the RT-DETR image processor."""

    images = [image for image, _, _ in batch]
    annotations = [annotation for _, annotation, _ in batch]
    image_ids = torch.tensor([annotation["image_id"] for _, annotation, _ in batch], dtype=torch.long)
    original_sizes = torch.tensor([size for _, _, size in batch], dtype=torch.long)
    encoded = image_processor(images=images, annotations=annotations, return_tensors="pt")
    encoded["image_ids"] = image_ids
    encoded["target_sizes"] = original_sizes
    return encoded


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], torch.Tensor]:
    """Move an encoded RT-DETR batch to the target device."""

    model_inputs = {
        "pixel_values": batch["pixel_values"].to(device),
    }
    if "pixel_mask" in batch:
        model_inputs["pixel_mask"] = batch["pixel_mask"].to(device)
    labels = [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in batch["labels"]
    ]
    target_sizes = batch["target_sizes"]
    return model_inputs, labels, target_sizes


def train_one_epoch(
    *,
    model: RTDetrForObjectDetection,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train RT-DETR for one epoch and return averaged losses."""

    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        model_inputs, labels, _ = move_batch_to_device(batch, device)
        outputs = model(**model_inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batches += 1
        totals["train_loss_total"] += float(loss.detach().cpu())
        if outputs.loss_dict is not None:
            for key, value in outputs.loss_dict.items():
                totals[f"train_{key}"] += float(value.detach().cpu())
    if batches == 0:
        return {}
    return {key: value / batches for key, value in totals.items()}


@torch.inference_mode()
def evaluate_model(
    *,
    model: RTDetrForObjectDetection,
    image_processor: Any,
    loader: DataLoader,
    annotation_path: Path,
    class_names: list[str],
    output_path: Path,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate RT-DETR on one split and return harmonized metrics."""

    model.eval()
    detections: list[dict[str, Any]] = []
    for batch in loader:
        model_inputs, _, target_sizes = move_batch_to_device(batch, device)
        outputs = model(**model_inputs)
        results = image_processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=target_sizes,
        )
        for result, image_id in zip(results, batch["image_ids"], strict=True):
            detections.extend(format_detections(image_id=image_id, result=result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, detections)
    metrics = evaluate_coco_detections(
        annotation_path=annotation_path,
        detections=detections,
        class_names=class_names,
    )
    return metrics, detections


def format_detections(*, image_id: int | torch.Tensor, result: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Convert RT-DETR predictions to COCO detection records."""

    resolved_image_id = int(image_id.item()) if isinstance(image_id, torch.Tensor) else int(image_id)
    records: list[dict[str, Any]] = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"], strict=True):
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
        records.append(
            {
                "image_id": resolved_image_id,
                "category_id": int(label.detach().cpu()) + 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score.detach().cpu()),
            }
        )
    return records


def save_checkpoint(
    path: Path,
    *,
    model: RTDetrForObjectDetection,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    job: dict[str, Any],
) -> None:
    """Save one reproducible RT-DETR checkpoint."""

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "job": job,
        },
        path,
    )


def load_model_for_eval(
    *,
    model_name: str,
    checkpoint_path: Path,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    device: torch.device,
) -> RTDetrForObjectDetection:
    """Load the best RT-DETR checkpoint into a fresh model."""

    model = RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def build_epoch_row(
    *,
    epoch: int,
    train_losses: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
    epoch_time_seconds: float,
    best_checkpoint: bool,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    """Build one standardized epoch metric row."""

    return {
        "epoch": epoch,
        **train_losses,
        "val_map50": val_metrics.get("mAP50"),
        "val_map50_95": val_metrics.get("mAP50_95"),
        "val_precision": val_metrics.get("precision"),
        "val_recall": val_metrics.get("recall"),
        "AP50_torre": val_metrics.get("AP50_torre"),
        "AP50_95_torre": val_metrics.get("AP50_95_torre"),
        "Recall_torre": val_metrics.get("Recall_torre"),
        "Precision_torre": val_metrics.get("Precision_torre"),
        "AP50_isoladores": val_metrics.get("AP50_isoladores"),
        "AP50_95_isoladores": val_metrics.get("AP50_95_isoladores"),
        "Recall_isoladores": val_metrics.get("Recall_isoladores"),
        "Precision_isoladores": val_metrics.get("Precision_isoladores"),
        "learning_rate": learning_rate,
        "epoch_time_seconds": epoch_time_seconds,
        "best_checkpoint": best_checkpoint,
        "checkpoint_path": checkpoint_path,
    }


def write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write RT-DETR epoch metrics to CSV."""

    base_fields = [
        "epoch",
        "train_loss_total",
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
    extra_fields = sorted({key for row in rows for key in row if key not in base_fields})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base_fields, *extra_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in writer.fieldnames or []})


def should_stop_early(
    *,
    epoch: int,
    epochs_without_improvement: int,
    training: dict[str, Any],
) -> bool:
    """Apply symmetric early stopping with a mandatory minimum epoch count."""

    if not bool(training["early_stopping"]):
        return False
    if epoch < int(training["min_epochs"]):
        return False
    return epochs_without_improvement >= int(training["patience"])


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers deterministically from torch initial seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """CLI entrypoint for one RT-DETR benchmark job."""

    args = parse_args()
    result = run_job(load_job(args.job))
    print(Path(args.job).as_posix())
    print(result["status"])
    print(result["best_checkpoint_path"])


if __name__ == "__main__":
    main()
