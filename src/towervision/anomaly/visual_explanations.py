"""Visual explanations for anomaly benchmark models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from towervision.anomaly.backends.common import (
    _load_resized_image,
    imagenet_normalize,
    load_device,
    load_job,
    load_split_rows,
)
from towervision.anomaly.backends.cutpaste_backend import CutPasteModel
from towervision.utils.io import clean_directory, ensure_dir, read_json, write_json
from towervision.utils.viz import draw_anomaly_heatmap_overlay, render_contact_sheet


class CutPasteGradCam:
    """Grad-CAM helper for the CutPaste encoder."""

    def __init__(self, model: CutPasteModel) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = self.model.encoder.layer4[-1].conv2.register_forward_hook(self._forward_hook)

    def close(self) -> None:
        self._handle.remove()

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output
        self.gradients = None
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def score_and_cam(
        self,
        image_batch: torch.Tensor,
        *,
        mean_vector: torch.Tensor,
        std_vector: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        """Return the ROI score and a Grad-CAM map resized to the input image."""

        self.model.zero_grad(set_to_none=True)
        embedding = self.model.encode(image_batch)
        z_scores = (embedding - mean_vector.unsqueeze(0)) / std_vector.unsqueeze(0)
        score = torch.sqrt((z_scores * z_scores).mean(dim=1))
        score.sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=image_batch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return float(score.detach().cpu().item()), cam[0, 0].detach().cpu().numpy()


def render_cutpaste_visual_explanations(
    *,
    runs_root: Path,
    selected_seeds: list[int] | None = None,
    split_name: str = "test",
    top_k: int = 24,
) -> list[dict[str, Any]]:
    """Render Grad-CAM explanations for CutPaste runs."""

    results: list[dict[str, Any]] = []
    seed_filter = set(selected_seeds or [])
    for job_path in sorted((runs_root / "cutpaste").glob("seed_*/job.json")):
        seed = int(job_path.parent.name.removeprefix("seed_"))
        if seed_filter and seed not in seed_filter:
            continue
        results.append(
            render_cutpaste_explanations_for_job(
                job_path,
                split_name=split_name,
                top_k=top_k,
            )
        )
    return results


def render_cutpaste_explanations_for_job(
    job_path: Path,
    *,
    split_name: str = "test",
    top_k: int = 24,
) -> dict[str, Any]:
    """Render Grad-CAM explanations for one CutPaste job."""

    job = load_job(job_path)
    run_dir = Path(str(job["run_dir"]))
    output_root = clean_directory(run_dir / "visual_explanations" / split_name)
    images_dir = ensure_dir(output_root / "images")
    rows = load_split_rows(job, split_name)
    checkpoint = _load_cutpaste_checkpoint(run_dir)
    threshold_payload = read_json(run_dir / "threshold_selection.json", default={}) or {}
    threshold = float(threshold_payload.get("threshold", 0.0))
    model = CutPasteModel()
    model.load_state_dict(checkpoint["model_state"])
    device = load_device()
    model.to(device)
    model.eval()
    grad_cam = CutPasteGradCam(model)
    mean_vector = checkpoint["mean_vector"].to(device)
    std_vector = checkpoint["std_vector"].to(device).clamp_min(1e-6)
    input_size = int(job["training"]["input_size"])

    try:
        explanation_rows: list[dict[str, Any]] = []
        items_by_roi: dict[str, tuple[Path, str]] = {}
        anomaly_items: list[tuple[Path, str]] = []
        score_values: list[float] = []

        for row in rows:
            image_tensor = _load_resized_image(Path(row.crop_path), input_size=input_size)
            image_batch = imagenet_normalize(image_tensor.unsqueeze(0).to(device))
            score, cam = grad_cam.score_and_cam(
                image_batch,
                mean_vector=mean_vector,
                std_vector=std_vector,
            )
            score_values.append(score)
            predicted_label = int(score >= threshold)
            output_path = images_dir / f"{row.roi_id}__gradcam.png"
            draw_anomaly_heatmap_overlay(
                Path(row.crop_path),
                cam,
                output_path=output_path,
                mask_path=Path(row.mask_path) if row.mask_path else None,
            )
            label_lines = [
                row.roi_id,
                f"score={score:.4f} pred={predicted_label} label={row.label}",
                row.anomaly_type or row.source_kind or "normal",
            ]
            item = (output_path, "\n".join(label_lines))
            items_by_roi[row.roi_id] = item
            if row.label == 1:
                anomaly_items.append(item)
            explanation_rows.append(
                {
                    "roi_id": row.roi_id,
                    "image_id": row.image_id,
                    "split": row.split,
                    "label": row.label,
                    "score": score,
                    "prediction": predicted_label,
                    "threshold": threshold,
                    "crop_path": row.crop_path,
                    "generator_family": row.generator_family,
                    "anomaly_type": row.anomaly_type,
                    "severity": row.severity,
                    "explanation_method": "grad_cam",
                    "explanation_path": output_path.as_posix(),
                }
            )
    finally:
        grad_cam.close()

    explanation_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    _write_explanation_index_csv(output_root / "explanation_index.csv", explanation_rows)

    top_items = [
        items_by_roi[row["roi_id"]]
        for row in explanation_rows[:top_k]
        if row["roi_id"] in items_by_roi
    ]
    top_sheet_path = None
    if top_items:
        top_sheet_path = (output_root / "contact_sheet_top_scores.png").as_posix()
        render_contact_sheet(
            top_items,
            output_path=Path(top_sheet_path),
            columns=4,
            title=f"CutPaste seed {job['seed']} Grad-CAM top scores ({split_name})",
        )

    anomaly_sheet_path = None
    if anomaly_items:
        anomaly_sheet_path = (output_root / "contact_sheet_anomalies.png").as_posix()
        render_contact_sheet(
            anomaly_items,
            output_path=Path(anomaly_sheet_path),
            columns=4,
            title=f"CutPaste seed {job['seed']} Grad-CAM anomalies ({split_name})",
        )

    summary = {
        "model_name": str(job["model"]["name"]),
        "display_name": str(job["model"]["display_name"]),
        "seed": int(job["seed"]),
        "backend": str(job["model"]["backend"]),
        "supported": True,
        "split": split_name,
        "explanation_method": "grad_cam",
        "checkpoint_path": (run_dir / "weights" / "best.pt").as_posix(),
        "output_root": output_root.as_posix(),
        "explanation_count": len(explanation_rows),
        "top_scores_contact_sheet_path": top_sheet_path,
        "anomalies_contact_sheet_path": anomaly_sheet_path,
        "top_k": top_k,
        "score_min": float(min(score_values)) if score_values else 0.0,
        "score_max": float(max(score_values)) if score_values else 0.0,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def _load_cutpaste_checkpoint(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "weights" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing CutPaste checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    return payload


def _write_explanation_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "roi_id",
        "image_id",
        "split",
        "label",
        "score",
        "prediction",
        "threshold",
        "crop_path",
        "generator_family",
        "anomaly_type",
        "severity",
        "explanation_method",
        "explanation_path",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
