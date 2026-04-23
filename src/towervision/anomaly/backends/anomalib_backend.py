"""Anomalib-backed runner for PatchCore and PaDiM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils import Split
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from torch.utils.data import DataLoader

from towervision.anomaly.backends.common import (
    build_anomalib_preprocessor,
    build_score_rows,
    build_threshold_payload,
    compute_breakdowns,
    compute_split_metrics,
    flatten_anomalib_predictions,
    flatten_rows_scores,
    load_job,
    load_split_rows,
    persist_run_outputs,
    set_random_seed,
    timer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Path to the anomaly benchmark job spec.")
    return parser.parse_args()


def _build_folder_dataset(
    *,
    split_root: Path,
    split_name: str,
    input_size: int,
    include_masks: bool,
) -> FolderDataset:
    normal_dir = split_root / "normal"
    abnormal_dir = split_root / "anomaly"
    mask_dir = split_root / "masks" if include_masks else None
    if split_name == "train":
        return FolderDataset(
            name="tower_vision",
            normal_dir=normal_dir,
            split=Split.TRAIN,
        )
    return FolderDataset(
        name="tower_vision",
        normal_dir=normal_dir,
        normal_test_dir=normal_dir,
        abnormal_dir=abnormal_dir if abnormal_dir.exists() else None,
        mask_dir=mask_dir if mask_dir and mask_dir.exists() else None,
        split=Split.TEST,
    )


def _make_loader(dataset: FolderDataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )


def _build_model(job: dict[str, Any]) -> torch.nn.Module:
    model_name = str(job["model"]["name"])
    extra = dict(job["model"].get("extra", {}))
    pre_processor = build_anomalib_preprocessor(int(job["training"]["input_size"]))
    common_kwargs = {
        "pre_processor": pre_processor,
        "post_processor": True,
        "evaluator": False,
        "visualizer": False,
    }
    if model_name == "patchcore":
        return Patchcore(
            backbone=str(extra.get("backbone", "resnet18")),
            layers=list(extra.get("layers", ["layer2", "layer3"])),
            pre_trained=bool(extra.get("pre_trained", True)),
            coreset_sampling_ratio=float(extra.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(extra.get("num_neighbors", 9)),
            **common_kwargs,
        )
    if model_name == "padim":
        n_features = extra.get("n_features")
        return Padim(
            backbone=str(extra.get("backbone", "resnet18")),
            layers=list(extra.get("layers", ["layer1", "layer2", "layer3"])),
            pre_trained=bool(extra.get("pre_trained", True)),
            n_features=None if n_features in ("", None) else int(n_features),
            **common_kwargs,
        )
    raise ValueError(f"unsupported anomalib model: {model_name}")


def _prediction_index(prediction_batches: list[Any]) -> dict[str, Any]:
    flattened = flatten_anomalib_predictions(prediction_batches)
    index: dict[str, Any] = {}
    for item in flattened:
        raw_path = Path(item.image_path)
        index[item.image_path] = item
        index[raw_path.as_posix()] = item
        try:
            index[raw_path.resolve().as_posix()] = item
        except FileNotFoundError:
            continue
    return index


def run_anomalib_job(job_path: Path) -> dict[str, Any]:
    """Execute one PatchCore or PaDiM benchmark job."""

    job = load_job(job_path)
    run_dir = Path(str(job["run_dir"]))
    set_random_seed(int(job["seed"]))
    torch.set_float32_matmul_precision("high")
    model = _build_model(job)
    extra = dict(job["model"].get("extra", {}))
    batch_size = int(extra.get("batch_size", 8))
    num_workers = int(extra.get("num_workers", 4))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    split_roots = {
        split_name: Path(path)
        for split_name, path in dict(job["dataset_views"]["split_root_dirs"]).items()
    }

    train_rows = load_split_rows(job, "train")
    val_rows = load_split_rows(job, "val")
    test_rows = load_split_rows(job, "test")

    train_loader = _make_loader(
        _build_folder_dataset(
            split_root=split_roots["train"],
            split_name="train",
            input_size=int(job["training"]["input_size"]),
            include_masks=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = _make_loader(
        _build_folder_dataset(
            split_root=split_roots["val"],
            split_name="val",
            input_size=int(job["training"]["input_size"]),
            include_masks=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = _make_loader(
        _build_folder_dataset(
            split_root=split_roots["test"],
            split_name="test",
            input_size=int(job["training"]["input_size"]),
            include_masks=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    start = timer()
    engine = Engine(
        default_root_dir=run_dir,
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    engine.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    fit_duration = timer() - start

    val_prediction_batches = engine.predict(model, dataloaders=val_loader, return_predictions=True)
    test_prediction_batches = engine.predict(model, dataloaders=test_loader, return_predictions=True)
    val_score_by_path = _prediction_index(val_prediction_batches or [])
    test_score_by_path = _prediction_index(test_prediction_batches or [])

    val_labels, val_scores = flatten_rows_scores(val_rows, score_by_path=val_score_by_path)
    threshold_payload = build_threshold_payload(
        val_labels,
        val_scores,
        recall_floor=float(job["ranking"]["operating_recall_floor"]),
    )
    threshold = float(threshold_payload["threshold"])
    val_metrics = compute_split_metrics(val_rows, score_by_path=val_score_by_path, threshold=threshold)
    test_metrics = compute_split_metrics(test_rows, score_by_path=test_score_by_path, threshold=threshold)
    val_score_rows = build_score_rows(val_rows, score_by_path=val_score_by_path, threshold=threshold)
    test_score_rows = build_score_rows(test_rows, score_by_path=test_score_by_path, threshold=threshold)
    breakdowns = compute_breakdowns(test_score_rows, threshold=threshold)

    checkpoint_callback = getattr(engine.trainer, "checkpoint_callback", None)
    best_checkpoint_path = ""
    if checkpoint_callback is not None:
        best_checkpoint_path = str(getattr(checkpoint_callback, "best_model_path", "") or "")

    train_history = [
        {
            "epoch": 1.0,
            "train_loss_total": 0.0,
            "val_roi_auroc": float(val_metrics["roi_auroc"]),
            "val_roi_auprc": float(val_metrics["roi_auprc"]),
            "val_f1": float(val_metrics["f1"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "learning_rate": 0.0,
            "epoch_time_seconds": fit_duration,
            "is_best": 1.0,
        }
    ]
    model_payload = {
        "model_name": str(job["model"]["name"]),
        "display_name": str(job["model"]["display_name"]),
        "backend": "anomalib",
        "fit_mode": str(job["model"]["fit_mode"]),
        "implementation_status": "reference_via_anomalib",
        "anomalib_class": type(model).__name__,
        "checkpoint_path": best_checkpoint_path,
        "notes": [
            "PatchCore/PaDiM executados via anomalib 2.2.0.",
        ],
    }
    return persist_run_outputs(
        run_dir=run_dir,
        model_name=str(job["model"]["name"]),
        display_name=str(job["model"]["display_name"]),
        backend="anomalib",
        fit_mode=str(job["model"]["fit_mode"]),
        seed=int(job["seed"]),
        model_payload=model_payload,
        train_history=train_history,
        threshold_payload=threshold_payload,
        val_score_rows=val_score_rows,
        test_score_rows=test_score_rows,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        breakdowns=breakdowns,
        notes=[f"checkpoint_path={best_checkpoint_path}" if best_checkpoint_path else "checkpoint_path="],
    )


def main() -> None:
    args = _parse_args()
    result = run_anomalib_job(args.job)
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
