"""YOLOX backend for the fair detection benchmark."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.hub import load_state_dict_from_url

from towervision.detectors.backends.coco_eval import evaluate_coco_detections
from towervision.utils.io import read_json, write_json


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


class FairYOLOXTrainer:
    """Single-GPU YOLOX trainer with benchmark-specific checkpointing and early stopping."""

    def __init__(
        self,
        exp: Any,
        args: SimpleNamespace,
        *,
        job: dict[str, Any],
        class_names: list[str],
        run_dir: Path,
    ) -> None:
        from yolox.core.trainer import Trainer

        self._trainer = Trainer(exp, args)
        self.job = job
        self.class_names = class_names
        self.run_dir = run_dir
        self.max_epoch = exp.max_epoch
        self.patience = int(job["training"]["patience"])
        self.min_epochs = int(job["training"]["min_epochs"])
        self.best_metric = float("-inf")
        self.best_epoch: int | None = None
        self.best_val_metrics: dict[str, float] | None = None
        self.current_val_metrics: dict[str, float] | None = None
        self.should_stop = False
        self.epoch_rows: list[dict[str, Any]] = []
        self._epoch_loss_sums: dict[str, float] = {}
        self._epoch_loss_count = 0
        self._epoch_started_at = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trainer, name)

    def train(self) -> None:
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as error:
            logger.error("Exception in training: {}", error)
            raise
        finally:
            self.after_train()

    def before_train(self) -> None:
        self._trainer.before_train()

    def after_train(self) -> None:
        self._trainer.after_train()

    def train_in_epoch(self) -> None:
        for epoch in range(self.start_epoch, self.max_epoch):
            self.epoch = epoch
            self._trainer.epoch = epoch
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
            if self.should_stop:
                break

    def before_epoch(self) -> None:
        self._epoch_started_at = time.perf_counter()
        self._epoch_loss_sums = {}
        self._epoch_loss_count = 0
        self._trainer.before_epoch()

    def train_in_iter(self) -> None:
        for iteration in range(self.max_iter):
            self.iter = iteration
            self._trainer.iter = iteration
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def before_iter(self) -> None:
        self._trainer.before_iter()

    def train_one_iter(self) -> None:
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        self._epoch_loss_count += 1
        for key, value in outputs.items():
            scalar = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            self._epoch_loss_sums[key] = self._epoch_loss_sums.get(key, 0.0) + scalar

    def after_iter(self) -> None:
        self._trainer.after_iter()

    def after_epoch(self) -> None:
        from yolox.utils import all_reduce_norm

        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
        if self.should_stop:
            logger.info(
                "Early stopping triggered at epoch {} with best epoch {}",
                self.epoch + 1,
                self.best_epoch,
            )

    def evaluate_and_save_model(self) -> None:
        from yolox.utils import adjust_status, is_parallel, synchronize

        if self.use_model_ema:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model
            if is_parallel(eval_model):
                eval_model = eval_model.module

        with adjust_status(eval_model, training=False):
            (_, _, summary), predictions = self.exp.eval(
                eval_model,
                self.evaluator,
                self.is_distributed,
                return_outputs=True,
            )

        detections = predictions_to_detections(predictions)
        metrics = evaluate_coco_detections(
            annotation_path=Path(self.exp.data_dir) / "annotations" / self.exp.val_ann,
            detections=detections,
            class_names=self.class_names,
        )
        self.current_val_metrics = metrics

        metric_value = float(metrics["mAP50_95"])
        improved = metric_value > self.best_metric
        if improved:
            self.best_metric = metric_value
            self.best_epoch = self.epoch + 1
            self.best_val_metrics = metrics
        self.best_ap = max(self.best_ap, metric_value)

        self.save_ckpt("last_epoch", improved, ap=metric_value)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=metric_value)

        logger.info("\n{}", summary)
        logger.info(
            "epoch={} val_map50_95={:.4f} best={:.4f} improved={}",
            self.epoch + 1,
            metric_value,
            self.best_metric,
            improved,
        )
        self.epoch_rows.append(
            build_epoch_row(
                epoch=self.epoch + 1,
                loss_sums=self._epoch_loss_sums,
                loss_count=self._epoch_loss_count,
                val_metrics=metrics,
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                epoch_time_seconds=time.perf_counter() - self._epoch_started_at,
                best_checkpoint=improved,
                checkpoint_path=(self.file_name + "/best_ckpt.pth") if improved else None,
            )
        )
        synchronize()

        if self.best_epoch is not None and (self.epoch + 1) >= self.min_epochs:
            no_improve = (self.epoch + 1) - self.best_epoch
            self.should_stop = no_improve >= self.patience

    def save_ckpt(self, ckpt_name: str, update_best_ckpt: bool = False, ap: float | None = None) -> None:
        self._trainer.save_ckpt(ckpt_name=ckpt_name, update_best_ckpt=update_best_ckpt, ap=ap)


def configure_exp(job: dict[str, Any], *, run_dir: Path) -> Any:
    """Create a YOLOX exp configured for the benchmark dataset and protocol."""

    from yolox.exp import get_exp

    exp = get_exp(exp_name="yolox-s")
    dataset_views = job["dataset_views"]
    training = job["training"]
    model_cfg = job["model"]

    exp.output_dir = run_dir.parent.as_posix()
    exp.exp_name = run_dir.name
    exp.num_classes = len(job["class_names"])
    exp.data_dir = dataset_views["coco_root"]
    exp.train_ann = "instances_train2017.json"
    exp.val_ann = "instances_val2017.json"
    exp.test_ann = "instances_test2017.json"
    exp.input_size = (int(training["img_size"]), int(training["img_size"]))
    exp.test_size = (int(training["img_size"]), int(training["img_size"]))
    exp.max_epoch = int(training["max_epochs"])
    exp.eval_interval = int(training["validate_every"])
    exp.seed = int(training["seed"])
    exp.data_num_workers = int(model_cfg["num_workers"])
    exp.multiscale_range = 0
    exp.mosaic_prob = 0.0
    exp.mixup_prob = 0.0
    exp.enable_mixup = False
    exp.flip_prob = float(training["augmentations"].get("horizontal_flip", 0.5))
    exp.hsv_prob = 0.5 if training["augmentations"].get("color_jitter") == "light" else 0.0
    exp.degrees = 0.0
    exp.translate = 0.05
    exp.shear = 0.0
    exp.test_conf = float(model_cfg.get("confidence_threshold", 0.001))
    exp.nmsthre = float(model_cfg.get("nms_iou_threshold", 0.6))
    return exp


def build_args(job: dict[str, Any], *, run_dir: Path, ckpt_path: Path | None) -> SimpleNamespace:
    """Build a minimal args namespace compatible with the official YOLOX trainer."""

    return SimpleNamespace(
        experiment_name=run_dir.name,
        name="yolox-s",
        dist_backend="nccl",
        dist_url=None,
        batch_size=int(job["model"]["batch_size"]),
        devices=1,
        exp_file=None,
        resume=False,
        ckpt=ckpt_path.as_posix() if ckpt_path is not None else None,
        start_epoch=None,
        num_machines=1,
        machine_rank=0,
        fp16=False,
        cache=None,
        occupy=False,
        logger="tensorboard",
        opts=[],
    )


def ensure_pretrained_checkpoint() -> Path:
    """Download the official YOLOX-s COCO checkpoint if needed and return its cache path."""

    from yolox.models.build import _CKPT_FULL_PATH

    checkpoint_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "yolox_s.pth"
    if checkpoint_path.exists():
        return checkpoint_path

    state_dict = load_state_dict_from_url(_CKPT_FULL_PATH["yolox-s"], map_location="cpu")
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one YOLOX benchmark job."""

    from yolox.tools.train import configure_module, configure_nccl, configure_omp

    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(job["class_names"])

    configure_module()
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    seed_everything(int(job["training"]["seed"]))

    exp = configure_exp(job, run_dir=run_dir)
    pretrained_checkpoint = ensure_pretrained_checkpoint()
    args = build_args(job, run_dir=run_dir, ckpt_path=pretrained_checkpoint)

    trainer = FairYOLOXTrainer(exp, args, job=job, class_names=class_names, run_dir=run_dir)
    trainer.train()

    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    write_epoch_metrics(epoch_metrics_path, trainer.epoch_rows)

    best_checkpoint = run_dir / "best_ckpt.pth"
    last_checkpoint = run_dir / "latest_ckpt.pth"
    best_val_metrics, val_predictions = evaluate_checkpoint(
        exp=exp,
        checkpoint_path=best_checkpoint,
        split="val",
        batch_size=int(job["model"]["batch_size"]),
        class_names=class_names,
    )
    write_json(run_dir / "val_eval" / "predictions.json", val_predictions)
    test_metrics, test_predictions = evaluate_checkpoint(
        exp=exp,
        checkpoint_path=best_checkpoint,
        split="test",
        batch_size=int(job["model"]["batch_size"]),
        class_names=class_names,
    )
    write_json(run_dir / "test_eval" / "predictions.json", test_predictions)

    result = {
        "model_name": job["model"]["name"],
        "display_name": job["model"]["display_name"],
        "seed": job["training"]["seed"],
        "status": "completed",
        "best_epoch": trainer.best_epoch,
        "best_checkpoint_path": best_checkpoint.as_posix() if best_checkpoint.exists() else None,
        "last_checkpoint_path": last_checkpoint.as_posix() if last_checkpoint.exists() else None,
        "train_log_path": (run_dir / "train.log").as_posix(),
        "epoch_metrics_path": epoch_metrics_path.as_posix(),
        "val_best_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "yolox backend executed with official package and custom early stopping by val_map50_95",
            "precision/recall are simple IoU50 matching metrics at score >= 0.001",
        ],
    }
    write_json(run_dir / "result.json", result)
    return result


def evaluate_checkpoint(
    *,
    exp: Any,
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    class_names: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate one checkpoint on val or test using YOLOX COCO evaluator."""

    from yolox.utils.checkpoint import load_ckpt

    model = exp.get_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = load_ckpt(model, checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    original_val_ann = exp.val_ann
    evaluator = exp.get_evaluator(
        batch_size=batch_size,
        is_distributed=False,
        testdev=(split == "test"),
    )
    (_, _, _), predictions = exp.eval(
        model,
        evaluator,
        False,
        return_outputs=True,
    )
    detections = predictions_to_detections(predictions)
    annotation_path = Path(exp.data_dir) / "annotations" / (
        exp.test_ann if split == "test" else exp.val_ann
    )
    metrics = evaluate_coco_detections(
        annotation_path=annotation_path,
        detections=detections,
        class_names=class_names,
    )
    exp.val_ann = original_val_ann
    return metrics, detections


def predictions_to_detections(predictions: dict[int, dict[str, list[Any]]]) -> list[dict[str, Any]]:
    """Convert YOLOX evaluator outputs to benchmark COCO detections."""

    detections: list[dict[str, Any]] = []
    for image_id, image_predictions in predictions.items():
        for bbox, score, category_id in zip(
            image_predictions.get("bboxes", []),
            image_predictions.get("scores", []),
            image_predictions.get("categories", []),
            strict=True,
        ):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    return detections


def build_epoch_row(
    *,
    epoch: int,
    loss_sums: dict[str, float],
    loss_count: int,
    val_metrics: dict[str, float],
    learning_rate: float,
    epoch_time_seconds: float,
    best_checkpoint: bool,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    """Build one standardized epoch metric row."""

    averaged_losses = {
        key: (value / max(loss_count, 1)) for key, value in loss_sums.items()
    }
    row = {
        "epoch": epoch,
        "train_loss_total": averaged_losses.get("total_loss"),
        "train_iou_loss": averaged_losses.get("iou_loss"),
        "train_conf_loss": averaged_losses.get("conf_loss"),
        "train_cls_loss": averaged_losses.get("cls_loss"),
        "train_l1_loss": averaged_losses.get("l1_loss"),
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
    for key, value in averaged_losses.items():
        row[f"train_{key}"] = value
    return row


def write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write YOLOX epoch metrics to CSV."""

    fieldnames = [
        "epoch",
        "train_loss_total",
        "train_iou_loss",
        "train_conf_loss",
        "train_cls_loss",
        "train_l1_loss",
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
    extra_fields = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in writer.fieldnames or []})


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """CLI entrypoint for one YOLOX benchmark job."""

    args = parse_args()
    result = run_job(load_job(args.job))
    print(Path(args.job).as_posix())
    print(result["status"])
    print(result["best_checkpoint_path"])


if __name__ == "__main__":
    main()
