"""Prepare and optionally execute the fair detection benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.detectors.benchmark import run_detection_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/det_benchmark_fair_v1.yaml",
        help="Benchmark experiment config.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=ROOT / "params.yaml",
        help="Active params.yaml.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the queued benchmark jobs instead of only preparing them.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of seeds.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark preparation or execution."""

    args = parse_args()
    result = run_detection_benchmark(
        repo_root=ROOT,
        params_path=args.params,
        benchmark_config_path=args.config,
        execute=args.execute,
        selected_models=args.models,
        selected_seeds=args.seeds,
    )

    print(f"dataset_materialized_dir={result['dataset_materialized_dir']}")
    print(f"job_index_path={result['job_index_path']}")
    print(f"runs_root={result['runs_root']}")
    for name, path in result["report_artifacts"].items():
        print(f"report_{name}={path}")


if __name__ == "__main__":
    main()
