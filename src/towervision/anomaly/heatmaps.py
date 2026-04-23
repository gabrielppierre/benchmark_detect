"""Render anomaly heatmaps from trained anomaly benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.engine import Engine

from towervision.anomaly.backends.anomalib_backend import (
    _build_folder_dataset,
    _build_model,
    _make_loader,
)
from towervision.anomaly.backends.common import (
    FlattenedPrediction,
    flatten_anomalib_predictions,
    load_job,
    load_split_rows,
)
from towervision.utils.io import clean_directory, ensure_dir, read_json, write_json
from towervision.utils.viz import draw_anomaly_heatmap_overlay, render_contact_sheet


def render_benchmark_heatmaps(
    *,
    runs_root: Path,
    selected_models: list[str] | None = None,
    selected_seeds: list[int] | None = None,
    split_name: str = "test",
    top_k: int = 24,
) -> list[dict[str, Any]]:
    """Render heatmaps for supported trained jobs under the anomaly benchmark root."""

    results: list[dict[str, Any]] = []
    for job_path in _iter_job_paths(
        runs_root,
        selected_models=selected_models,
        selected_seeds=selected_seeds,
    ):
        results.append(
            render_heatmaps_for_job(
                job_path,
                split_name=split_name,
                top_k=top_k,
            )
        )
    return results


def render_heatmaps_for_job(
    job_path: Path,
    *,
    split_name: str = "test",
    top_k: int = 24,
) -> dict[str, Any]:
    """Render heatmaps for one trained benchmark job."""

    job = load_job(job_path)
    run_dir = Path(str(job["run_dir"]))
    backend = str(job["model"]["backend"])
    output_root = clean_directory(run_dir / "anomaly_maps" / split_name)
    if backend != "anomalib":
        summary = {
            "model_name": str(job["model"]["name"]),
            "display_name": str(job["model"]["display_name"]),
            "seed": int(job["seed"]),
            "backend": backend,
            "supported": False,
            "reason": "heatmaps are only available for anomalib-backed models in the current implementation",
            "output_root": output_root.as_posix(),
        }
        write_json(output_root / "summary.json", summary)
        return summary

    model_payload = read_json(run_dir / "model.json", default={})
    checkpoint_path = Path(str(model_payload.get("checkpoint_path", "")))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint for heatmap rendering: {checkpoint_path}")

    extra = dict(job["model"].get("extra", {}))
    input_size = int(job["training"]["input_size"])
    batch_size = int(extra.get("batch_size", 8))
    num_workers = int(extra.get("num_workers", 4))
    split_root = Path(str(job["dataset_views"]["split_root_dirs"][split_name]))
    rows = load_split_rows(job, split_name)
    dataset = _build_folder_dataset(
        split_root=split_root,
        split_name=split_name,
        input_size=input_size,
        include_masks=True,
    )
    dataloader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    base_model = _build_model(job)
    model_class = type(base_model)
    model = model_class.load_from_checkpoint(checkpoint_path, weights_only=False)
    engine = Engine(
        default_root_dir=run_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    prediction_batches = engine.predict(
        model,
        dataloaders=dataloader,
        return_predictions=True,
    )
    flattened_predictions = flatten_anomalib_predictions(prediction_batches or [])
    prediction_index = _build_prediction_index(flattened_predictions)
    value_range = _compute_global_value_range(flattened_predictions)

    images_dir = ensure_dir(output_root / "images")
    index_rows: list[dict[str, Any]] = []
    item_by_roi_id: dict[str, tuple[Path, str]] = {}
    anomaly_items: list[tuple[Path, str]] = []
    for row in rows:
        prediction = _lookup_prediction(prediction_index, row.crop_path)
        anomaly_map = np.asarray(prediction.anomaly_map).squeeze() if prediction.anomaly_map is not None else None
        if anomaly_map is None or anomaly_map.ndim != 2:
            continue
        output_path = images_dir / f"{row.roi_id}__heatmap.png"
        draw_anomaly_heatmap_overlay(
            Path(row.crop_path),
            anomaly_map,
            output_path=output_path,
            value_range=value_range,
            mask_path=Path(row.mask_path) if row.mask_path else None,
        )
        score = float(prediction.pred_score)
        threshold = _resolve_threshold(run_dir, score=score)
        predicted_label = int(score >= threshold)
        label_lines = [
            row.roi_id,
            f"score={score:.4f} pred={predicted_label} label={row.label}",
            row.anomaly_type or row.source_kind or "normal",
        ]
        item_by_roi_id[row.roi_id] = (output_path, "\n".join(label_lines))
        if row.label == 1:
            anomaly_items.append((output_path, "\n".join(label_lines)))
        index_rows.append(
            {
                "roi_id": row.roi_id,
                "image_id": row.image_id,
                "split": row.split,
                "label": row.label,
                "score": score,
                "prediction": predicted_label,
                "threshold": threshold,
                "crop_path": row.crop_path,
                "mask_path": row.mask_path,
                "generator_family": row.generator_family,
                "anomaly_type": row.anomaly_type,
                "severity": row.severity,
                "heatmap_path": output_path.as_posix(),
            }
        )

    index_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    _write_heatmap_index_csv(output_root / "heatmap_index.csv", index_rows)

    top_items = [item_by_roi_id[row["roi_id"]] for row in index_rows[:top_k] if row["roi_id"] in item_by_roi_id]

    top_sheet_path = None
    if top_items:
        top_sheet_path = (output_root / "contact_sheet_top_scores.png").as_posix()
        render_contact_sheet(
            top_items,
            output_path=Path(top_sheet_path),
            columns=4,
            title=f"{job['model']['display_name']} seed {job['seed']} top scores ({split_name})",
        )

    anomaly_sheet_path = None
    if anomaly_items:
        anomaly_sheet_path = (output_root / "contact_sheet_anomalies.png").as_posix()
        render_contact_sheet(
            anomaly_items,
            output_path=Path(anomaly_sheet_path),
            columns=4,
            title=f"{job['model']['display_name']} seed {job['seed']} anomalies ({split_name})",
        )

    summary = {
        "model_name": str(job["model"]["name"]),
        "display_name": str(job["model"]["display_name"]),
        "seed": int(job["seed"]),
        "backend": backend,
        "supported": True,
        "split": split_name,
        "checkpoint_path": checkpoint_path.as_posix(),
        "output_root": output_root.as_posix(),
        "heatmap_count": len(index_rows),
        "normalization_low": value_range[0],
        "normalization_high": value_range[1],
        "top_scores_contact_sheet_path": top_sheet_path,
        "anomalies_contact_sheet_path": anomaly_sheet_path,
        "top_k": top_k,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def _iter_job_paths(
    runs_root: Path,
    *,
    selected_models: list[str] | None,
    selected_seeds: list[int] | None,
) -> list[Path]:
    paths: list[Path] = []
    allowed_models = set(selected_models or [])
    allowed_seeds = set(selected_seeds or [])
    for job_path in sorted(runs_root.glob("*/seed_*/job.json")):
        model_name = job_path.parent.parent.name
        seed = int(job_path.parent.name.removeprefix("seed_"))
        if allowed_models and model_name not in allowed_models:
            continue
        if allowed_seeds and seed not in allowed_seeds:
            continue
        paths.append(job_path)
    return paths


def _build_prediction_index(predictions: list[FlattenedPrediction]) -> dict[str, FlattenedPrediction]:
    result: dict[str, FlattenedPrediction] = {}
    for prediction in predictions:
        raw_path = Path(prediction.image_path)
        result[prediction.image_path] = prediction
        result[raw_path.as_posix()] = prediction
        try:
            result[raw_path.resolve().as_posix()] = prediction
        except FileNotFoundError:
            continue
    return result


def _lookup_prediction(
    prediction_index: dict[str, FlattenedPrediction],
    crop_path: str,
) -> FlattenedPrediction:
    path = Path(crop_path)
    candidates = [crop_path, path.as_posix()]
    try:
        candidates.append(path.resolve().as_posix())
    except FileNotFoundError:
        pass
    for candidate in candidates:
        prediction = prediction_index.get(candidate)
        if prediction is not None:
            return prediction
    raise KeyError(crop_path)


def _compute_global_value_range(predictions: list[FlattenedPrediction]) -> tuple[float, float]:
    maps: list[np.ndarray] = []
    for prediction in predictions:
        if prediction.anomaly_map is None:
            continue
        anomaly_map = np.asarray(prediction.anomaly_map).squeeze()
        if anomaly_map.ndim != 2:
            continue
        maps.append(anomaly_map.reshape(-1).astype(np.float32))
    if not maps:
        return 0.0, 1.0
    stacked = np.concatenate(maps, axis=0)
    low = float(np.quantile(stacked, 0.01))
    high = float(np.quantile(stacked, 0.99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(stacked))
        high = float(np.max(stacked))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0, 1.0
    return low, high


def _resolve_threshold(run_dir: Path, *, score: float) -> float:
    payload = read_json(run_dir / "threshold_selection.json", default={})
    raw_threshold = payload.get("threshold", score)
    return float(raw_threshold)


def _write_heatmap_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "roi_id",
        "image_id",
        "split",
        "label",
        "score",
        "prediction",
        "threshold",
        "crop_path",
        "mask_path",
        "generator_family",
        "anomaly_type",
        "severity",
        "heatmap_path",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
