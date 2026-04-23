"""Thin CLI wrapper for the anomaly benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.anomaly.benchmark import run_anomaly_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
        help="Path to params.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/anom_benchmark_fair_v1.yaml"),
        help="Path to the anomaly benchmark config",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the queued jobs instead of only preparing them",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to schedule",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of seeds to schedule",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_anomaly_benchmark(
        repo_root=ROOT,
        params_path=args.params,
        benchmark_config_path=args.config,
        execute=args.execute,
        selected_models=args.models,
        selected_seeds=args.seeds,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
