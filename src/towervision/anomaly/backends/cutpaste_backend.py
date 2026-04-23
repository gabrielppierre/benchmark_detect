"""Repository-local CutPaste backend for the anomaly benchmark."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from towervision.anomaly.backends.common import (
    RoiImageDataset,
    build_score_rows,
    build_threshold_payload,
    compute_breakdowns,
    compute_split_metrics,
    flatten_rows_scores,
    imagenet_normalize,
    load_device,
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


class CutPasteModel(nn.Module):
    """ResNet18 encoder with a small classifier head for CutPaste pretext training."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.classifier = nn.Linear(feature_dim, 3)
        self.feature_dim = feature_dim

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder(batch)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(batch)
        return self.classifier(embedding)


def _make_loader(
    rows: Sequence[Any],
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
        collate_fn=_collate_roi_samples,
    )


def _collate_roi_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    image_paths = [str(item["image_path"]) for item in batch]
    masks = [item["mask"] for item in batch]
    if any(mask is not None for mask in masks):
        mask_tensors = [
            mask if isinstance(mask, torch.Tensor) else torch.zeros(images.shape[-2:], dtype=torch.float32)
            for mask in masks
        ]
        batch_masks: torch.Tensor | None = torch.stack(mask_tensors, dim=0)
    else:
        batch_masks = None
    return {
        "image": images,
        "label": labels,
        "image_path": image_paths,
        "mask": batch_masks,
    }


def _sample_patch_shape(height: int, width: int, *, scar: bool) -> tuple[int, int]:
    area_ratio = random.uniform(0.02, 0.12)
    if scar:
        aspect_ratio = random.uniform(0.08, 0.25)
    else:
        aspect_ratio = random.uniform(0.3, 3.0)
    patch_area = max(4, int(area_ratio * height * width))
    patch_height = max(4, int((patch_area / aspect_ratio) ** 0.5))
    patch_width = max(4, int((patch_area * aspect_ratio) ** 0.5))
    return min(patch_height, height - 1), min(patch_width, width - 1)


def _cutpaste_one(image: torch.Tensor, *, scar: bool) -> torch.Tensor:
    cloned = image.clone()
    _, height, width = cloned.shape
    patch_height, patch_width = _sample_patch_shape(height, width, scar=scar)
    source_y = random.randint(0, max(0, height - patch_height))
    source_x = random.randint(0, max(0, width - patch_width))
    target_y = random.randint(0, max(0, height - patch_height))
    target_x = random.randint(0, max(0, width - patch_width))
    patch = cloned[:, source_y : source_y + patch_height, source_x : source_x + patch_width].clone()
    cloned[:, target_y : target_y + patch_height, target_x : target_x + patch_width] = patch
    return cloned


def _build_cutpaste_batch(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normal = images
    cutpaste = torch.stack([_cutpaste_one(image, scar=False) for image in images], dim=0)
    scar = torch.stack([_cutpaste_one(image, scar=True) for image in images], dim=0)
    training_batch = torch.cat([normal, cutpaste, scar], dim=0)
    labels = torch.cat(
        [
            torch.zeros(len(images), dtype=torch.long),
            torch.ones(len(images), dtype=torch.long),
            torch.full((len(images),), 2, dtype=torch.long),
        ],
        dim=0,
    )
    return training_batch, labels


def _collect_embeddings(
    loader: DataLoader,
    model: CutPasteModel,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[int], list[torch.Tensor | None]]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    image_paths: list[str] = []
    labels: list[int] = []
    masks: list[torch.Tensor | None] = []
    with torch.no_grad():
        for batch in loader:
            images = imagenet_normalize(batch["image"].to(device))
            features = model.encode(images)
            embeddings.append(features.detach().cpu())
            image_paths.extend(batch["image_path"])
            labels.extend(batch["label"].tolist())
            if batch["mask"] is not None:
                masks.extend([item.detach().cpu() for item in batch["mask"]])
            else:
                masks.extend([None] * len(batch["image_path"]))
    return torch.cat(embeddings, dim=0), image_paths, labels, masks


def _score_from_gaussian(
    embeddings: torch.Tensor,
    *,
    mean_vector: torch.Tensor,
    std_vector: torch.Tensor,
) -> torch.Tensor:
    z_scores = (embeddings - mean_vector.unsqueeze(0)) / std_vector.unsqueeze(0)
    return torch.sqrt((z_scores * z_scores).mean(dim=1))


def _prediction_index(
    rows: Sequence[Any],
    *,
    scores: torch.Tensor,
    image_paths: Sequence[str],
    labels: Sequence[int],
    masks: Sequence[torch.Tensor | None],
) -> dict[str, Any]:
    from towervision.anomaly.backends.common import FlattenedPrediction

    result: dict[str, Any] = {}
    for path, score, label, mask in zip(image_paths, scores.tolist(), labels, masks):
        result[str(path)] = FlattenedPrediction(
            image_path=str(path),
            pred_score=float(score),
            pred_label=0,
            gt_label=int(label),
            anomaly_map=None,
            gt_mask=mask.numpy() if isinstance(mask, torch.Tensor) else None,
        )
    return result


def run_cutpaste_job(job_path: Path) -> dict[str, Any]:
    """Execute one CutPaste benchmark job."""

    job = load_job(job_path)
    run_dir = Path(str(job["run_dir"]))
    set_random_seed(int(job["seed"]))
    torch.set_float32_matmul_precision("high")
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    extra = dict(job["model"].get("extra", {}))
    input_size = int(job["training"]["input_size"])
    batch_size = int(extra.get("batch_size", 16))
    num_workers = int(extra.get("num_workers", 4))
    learning_rate = float(extra.get("learning_rate", 1e-4))
    weight_decay = float(extra.get("weight_decay", 1e-5))
    device = load_device()

    train_rows = load_split_rows(job, "train")
    val_rows = load_split_rows(job, "val")
    test_rows = load_split_rows(job, "test")

    train_loader = _make_loader(
        train_rows,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    train_eval_loader = _make_loader(
        train_rows,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = _make_loader(
        val_rows,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = _make_loader(
        test_rows,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = CutPasteModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, Any] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_threshold_payload: dict[str, float] | None = None
    epochs_without_improvement = 0
    train_history: list[dict[str, float]] = []

    for epoch in range(1, int(job["training"]["max_epochs"]) + 1):
        start = timer()
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            train_batch, class_labels = _build_cutpaste_batch(batch["image"])
            train_batch = imagenet_normalize(train_batch.to(device))
            class_labels = class_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_batch)
            loss = criterion(logits, class_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        train_embeddings, _, _, _ = _collect_embeddings(train_eval_loader, model, device=device)
        mean_vector = train_embeddings.mean(dim=0)
        std_vector = train_embeddings.std(dim=0).clamp_min(1e-6)
        val_embeddings, val_paths, val_labels, val_masks = _collect_embeddings(val_loader, model, device=device)
        val_scores = _score_from_gaussian(val_embeddings, mean_vector=mean_vector, std_vector=std_vector)
        val_score_by_path = _prediction_index(
            val_rows,
            scores=val_scores,
            image_paths=val_paths,
            labels=val_labels,
            masks=val_masks,
        )
        val_label_values, val_score_values = flatten_rows_scores(val_rows, score_by_path=val_score_by_path)
        threshold_payload = build_threshold_payload(
            val_label_values,
            val_score_values,
            recall_floor=float(job["ranking"]["operating_recall_floor"]),
        )
        val_metrics = compute_split_metrics(
            val_rows,
            score_by_path=val_score_by_path,
            threshold=float(threshold_payload["threshold"]),
        )

        current_monitor = float(val_metrics["roi_auroc"])
        best_monitor = float(best_val_metrics["roi_auroc"]) if best_val_metrics is not None else float("-inf")
        is_best = current_monitor > best_monitor
        if is_best:
            best_val_metrics = val_metrics
            best_threshold_payload = threshold_payload
            best_state = {
                "model_state": model.state_dict(),
                "mean_vector": mean_vector.cpu(),
                "std_vector": std_vector.cpu(),
                "epoch": epoch,
            }
            torch.save(best_state, weights_dir / "best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        last_state = {
            "model_state": model.state_dict(),
            "mean_vector": mean_vector.cpu(),
            "std_vector": std_vector.cpu(),
            "epoch": epoch,
        }
        torch.save(last_state, weights_dir / "last.pt")

        epoch_time = timer() - start
        train_history.append(
            {
                "epoch": float(epoch),
                "train_loss_total": float(sum(losses) / len(losses)) if losses else 0.0,
                "val_roi_auroc": float(val_metrics["roi_auroc"]),
                "val_roi_auprc": float(val_metrics["roi_auprc"]),
                "val_f1": float(val_metrics["f1"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_time,
                "is_best": 1.0 if is_best else 0.0,
            }
        )
        if epoch >= int(job["training"]["min_epochs"]) and epochs_without_improvement >= int(job["training"]["patience"]):
            break

    if best_state is None or best_val_metrics is None or best_threshold_payload is None:
        raise RuntimeError("CutPaste training did not produce a valid checkpoint")

    model.load_state_dict(best_state["model_state"])
    mean_vector = best_state["mean_vector"].to(device)
    std_vector = best_state["std_vector"].to(device)
    threshold = float(best_threshold_payload["threshold"])

    val_embeddings, val_paths, val_labels, val_masks = _collect_embeddings(val_loader, model, device=device)
    val_scores = _score_from_gaussian(val_embeddings.to(device), mean_vector=mean_vector, std_vector=std_vector).cpu()
    val_score_by_path = _prediction_index(
        val_rows,
        scores=val_scores,
        image_paths=val_paths,
        labels=val_labels,
        masks=val_masks,
    )
    test_embeddings, test_paths, test_labels, test_masks = _collect_embeddings(test_loader, model, device=device)
    test_scores = _score_from_gaussian(test_embeddings.to(device), mean_vector=mean_vector, std_vector=std_vector).cpu()
    test_score_by_path = _prediction_index(
        test_rows,
        scores=test_scores,
        image_paths=test_paths,
        labels=test_labels,
        masks=test_masks,
    )

    val_metrics = compute_split_metrics(val_rows, score_by_path=val_score_by_path, threshold=threshold)
    test_metrics = compute_split_metrics(test_rows, score_by_path=test_score_by_path, threshold=threshold)
    val_score_rows = build_score_rows(val_rows, score_by_path=val_score_by_path, threshold=threshold)
    test_score_rows = build_score_rows(test_rows, score_by_path=test_score_by_path, threshold=threshold)
    breakdowns = compute_breakdowns(test_score_rows, threshold=threshold)

    model_payload = {
        "model_name": str(job["model"]["name"]),
        "display_name": str(job["model"]["display_name"]),
        "backend": "repo_cutpaste",
        "fit_mode": str(job["model"]["fit_mode"]),
        "implementation_status": "repo_native_cutpaste",
        "feature_dim": model.feature_dim,
        "best_checkpoint_path": (weights_dir / "best.pt").as_posix(),
        "last_checkpoint_path": (weights_dir / "last.pt").as_posix(),
        "notes": [
            "Repository-native CutPaste implementation with Gaussian scoring over learned embeddings.",
        ],
    }
    notes = [
        f"best_epoch={int(best_state['epoch'])}",
        f"best_checkpoint_path={(weights_dir / 'best.pt').as_posix()}",
    ]
    return persist_run_outputs(
        run_dir=run_dir,
        model_name=str(job["model"]["name"]),
        display_name=str(job["model"]["display_name"]),
        backend="repo_cutpaste",
        fit_mode=str(job["model"]["fit_mode"]),
        seed=int(job["seed"]),
        model_payload=model_payload,
        train_history=train_history,
        threshold_payload=best_threshold_payload,
        val_score_rows=val_score_rows,
        test_score_rows=test_score_rows,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        breakdowns=breakdowns,
        notes=notes,
    )


def main() -> None:
    args = _parse_args()
    result = run_cutpaste_job(args.job)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
