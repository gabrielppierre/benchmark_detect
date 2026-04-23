"""Placeholder anomaly inference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from towervision.utils.io import write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class AnomalyScore:
    """Anomaly score for one ROI crop."""

    crop_path: Path
    score: float
    label: int | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "crop_path": self.crop_path.as_posix(),
            "score": self.score,
            "label": self.label,
            "source": self.source,
        }


def score_crop_name(crop_path: Path) -> tuple[float, int | None]:
    """Produce a deterministic score from the crop file name."""

    name = crop_path.stem.lower()
    if "anom" in name or "defect" in name:
        return 0.9, 1
    if "normal" in name or "good" in name or "ok" in name:
        return 0.1, 0
    return ((len(name) % 7) + 2) / 10, None


def infer_anomaly_scores(
    crops_dir: Path,
    *,
    source: str,
) -> list[AnomalyScore]:
    """Score all image crops under a directory."""

    scores: list[AnomalyScore] = []
    for crop_path in sorted(crops_dir.glob("*")):
        if not crop_path.is_file() or crop_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        score, label = score_crop_name(crop_path)
        scores.append(AnomalyScore(crop_path=crop_path, score=score, label=label, source=source))
    return scores


def save_anomaly_scores(path: Path, scores: Iterable[AnomalyScore]) -> None:
    """Persist anomaly scores to JSON."""

    write_json(path, [score.to_dict() for score in scores])
