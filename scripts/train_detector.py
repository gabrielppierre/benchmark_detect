"""Train a placeholder detector artifact."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.detectors.train import train_detector  # noqa: E402
from towervision.utils.io import read_yaml  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    params = read_yaml(ROOT / "params.yaml")
    detector_name = params["detector"]["name"]
    detector_config = read_yaml(ROOT / f"configs/detector/{detector_name}.yaml")
    detector_config.update(params["detector"])

    train_detector(
        detector_config,
        split_path=_resolve_repo_path(params["paths"]["splits_path"]),
        output_path=_resolve_repo_path(params["paths"]["detector_run_dir"]) / "model.json",
    )


if __name__ == "__main__":
    main()
