"""Render anomaly heatmaps for trained anomaly benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.anomaly.heatmaps import render_benchmark_heatmaps
from towervision.utils.io import read_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of anomaly models to render. Defaults to all trained jobs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of seeds to render. Defaults to all trained jobs.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="Benchmark split to render. Defaults to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of highest-score ROIs to include in the top-scores contact sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = read_yaml(ROOT / "params.yaml", default={}) or {}
    runs_root = ROOT / str(params["paths"]["anomaly_benchmark_run_root"])
    results = render_benchmark_heatmaps(
        runs_root=runs_root,
        selected_models=list(args.models) if args.models else None,
        selected_seeds=list(args.seeds) if args.seeds else None,
        split_name=str(args.split),
        top_k=int(args.top_k),
    )
    for result in results:
        print(f"model={result['model_name']}")
        print(f"seed={result['seed']}")
        print(f"supported={result['supported']}")
        print(f"output_root={result['output_root']}")
        if result["supported"]:
            print(f"heatmap_count={result['heatmap_count']}")
            print(f"top_scores_contact_sheet={result['top_scores_contact_sheet_path']}")
            print(f"anomalies_contact_sheet={result['anomalies_contact_sheet_path']}")
        else:
            print(f"reason={result['reason']}")
        print()


if __name__ == "__main__":
    main()
