"""TorchVision Faster R-CNN backend for the fair detection benchmark."""

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
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as transform_functional

from towervision.detectors.backends.coco_eval import evaluate_coco_detections
from towervision.utils.io import read_json, write_json


class CocoDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Minimal COCO detection dataset for TorchVision models."""

    def __init__(
        self,
        *,
        image_dir: Path,
        annotation_path: Path,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        payload = read_json(annotation_path, default={})
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.images = list(payload.get("images", []))
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.horizontal_flip_prob = horizontal_flip_prob
        for annotation in payload.get("annotations", []):
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.image_dir / str(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            x, y, bbox_width, bbox_height = [float(value) for value in annotation["bbox"]]
            if bbox_width <= 0.0 or bbox_height <= 0.0:
                continue
            boxes.append([x, y, x + bbox_width, y + bbox_height])
            labels.append(int(annotation["category_id"]))
            areas.append(float(annotation.get("area", bbox_width * bbox_height)))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape((-1, 4))
        if self.horizontal_flip_prob > 0.0 and random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if box_tensor.numel() > 0:
                flipped = box_tensor.clone()
                flipped[:, 0] = width - box_tensor[:, 2]
                flipped[:, 2] = width - box_tensor[:, 0]
                box_tensor = flipped

        target = {
            "boxes": box_tensor,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return transform_functional.to_tensor(image), target


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
    """Execute one Faster R-CNN benchmark job."""

    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = job["model"]
    training = job["training"]
    dataset_views = job["dataset_views"]
    class_names = list(job["class_names"])
    seed = int(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)
    model = build_model(
        num_classes=len(class_names) + 1,
        img_size=int(training["img_size"]),
        pretrained=True,
    )
    model.to(device)

    train_loader = build_loader(job=job, split="train", train=True)
    val_loader = build_loader(job=job, split="val", train=False)
    test_loader = build_loader(job=job, split="test", train=False)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(model_cfg.get("extra", {}).get("lr", 0.005)),
        momentum=float(model_cfg.get("extra", {}).get("momentum", 0.9)),
        weight_decay=float(model_cfg.get("extra", {}).get("weight_decay", 0.0005)),
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
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            annotation_path=Path(dataset_views["coco_annotations"]["val"]),
            output_path=run_dir / "val_predictions_latest.json",
            class_names=class_names,
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
        epoch_time = time.perf_counter() - started_at
        row = build_epoch_row(
            epoch=epoch,
            train_losses=train_losses,
            val_metrics=val_metrics,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            epoch_time_seconds=epoch_time,
            best_checkpoint=improved,
            checkpoint_path=best_checkpoint.as_posix() if improved else None,
        )
        epoch_rows.append(row)
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
    final_val_metrics = evaluate_model(
        model=load_model_for_eval(best_checkpoint, len(class_names) + 1, int(training["img_size"]), device),
        loader=val_loader,
        annotation_path=Path(dataset_views["coco_annotations"]["val"]),
        output_path=run_dir / "val_eval" / "predictions.json",
        class_names=class_names,
        device=device,
    )
    test_metrics = evaluate_model(
        model=load_model_for_eval(best_checkpoint, len(class_names) + 1, int(training["img_size"]), device),
        loader=test_loader,
        annotation_path=Path(dataset_views["coco_annotations"]["test"]),
        output_path=run_dir / "test_eval" / "predictions.json",
        class_names=class_names,
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
        "val_best_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "torchvision backend executed with val mAP50-95 checkpoint selection",
            "precision/recall are simple IoU50 matching metrics at score >= 0.001",
        ],
    }
    write_json(run_dir / "result.json", result)
    return result


def build_model(*, num_classes: int, img_size: int, pretrained: bool) -> torch.nn.Module:
    """Build Faster R-CNN ResNet50-FPN v2 with a project-specific head."""

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=img_size,
        max_size=img_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_loader(*, job: dict[str, Any], split: str, train: bool) -> DataLoader:
    """Build a deterministic data loader for one split."""

    model_cfg = job["model"]
    training = job["training"]
    dataset_views = job["dataset_views"]
    horizontal_flip = float(training["augmentations"].get("horizontal_flip", 0.0)) if train else 0.0
    dataset = CocoDetectionDataset(
        image_dir=Path(dataset_views["coco_images"][split]),
        annotation_path=Path(dataset_views["coco_annotations"][split]),
        horizontal_flip_prob=horizontal_flip,
    )
    generator = torch.Generator()
    generator.manual_seed(int(training["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(model_cfg["num_workers"]),
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch and return mean loss components."""

    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batches += 1
        totals["train_loss_total"] += float(loss.detach().cpu())
        for key, value in loss_dict.items():
            totals[f"train_{key}"] += float(value.detach().cpu())
    if batches == 0:
        return {}
    return {key: value / batches for key, value in totals.items()}


@torch.inference_mode()
def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    annotation_path: Path,
    output_path: Path,
    class_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Run model inference and evaluate detections with COCO metrics."""

    model.eval()
    detections: list[dict[str, Any]] = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for target, output in zip(targets, outputs, strict=True):
            image_id = int(target["image_id"].item())
            detections.extend(format_detections(image_id=image_id, output=output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, detections)
    return evaluate_coco_detections(
        annotation_path=annotation_path,
        detections=detections,
        class_names=class_names,
    )


def format_detections(*, image_id: int, output: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Convert TorchVision predictions to COCO detection records."""

    records: list[dict[str, Any]] = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"], strict=True):
        category_id = int(label.detach().cpu())
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
        records.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score.detach().cpu()),
            }
        )
    return records


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    job: dict[str, Any],
) -> None:
    """Save a reproducible TorchVision checkpoint."""

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
    checkpoint_path: Path,
    num_classes: int,
    img_size: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load the best checkpoint into a fresh model for evaluation."""

    model = build_model(num_classes=num_classes, img_size=img_size, pretrained=False)
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
    """Write epoch metrics to CSV with stable base columns and dynamic losses."""

    base_fields = [
        "epoch",
        "train_loss_total",
        "train_loss_classifier",
        "train_loss_box_reg",
        "train_loss_objectness",
        "train_loss_rpn_box_reg",
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


def collate_detection_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple:
    """Collate variable-size detection targets."""

    return tuple(zip(*batch, strict=True))


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
    """CLI entrypoint for one TorchVision benchmark job."""

    args = parse_args()
    result = run_job(load_job(args.job))
    print(Path(args.job).as_posix())
    print(result["status"])
    print(result["best_checkpoint_path"])


if __name__ == "__main__":
    main()
