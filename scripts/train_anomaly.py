"""Train a placeholder anomaly model artifact."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.anomaly.train import train_anomaly_model  # noqa: E402
from towervision.utils.io import read_yaml  # noqa: E402


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    params = read_yaml(ROOT / "params.yaml")
    anomaly_name = params["anomaly"]["name"]
    anomaly_config = read_yaml(ROOT / f"configs/anomaly/{anomaly_name}.yaml")
    anomaly_config.update(params["anomaly"])

    crop_source = params["anomaly"]["crop_source"]
    source_dir_key = "gt_crops_dir" if crop_source == "gt_crops" else "pred_crops_dir"

    train_anomaly_model(
        anomaly_config,
        crops_dir=_resolve_repo_path(params["paths"][source_dir_key]),
        output_path=_resolve_repo_path(params["paths"]["anomaly_run_dir"]) / "model.json",
    )


if __name__ == "__main__":
    main()
