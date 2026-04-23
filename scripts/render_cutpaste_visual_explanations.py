"""Render Grad-CAM visual explanations for trained CutPaste runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.anomaly.visual_explanations import render_cutpaste_visual_explanations
from towervision.utils.io import read_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of CutPaste seeds to render. Defaults to all available seeds.",
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
    results = render_cutpaste_visual_explanations(
        runs_root=runs_root,
        selected_seeds=list(args.seeds) if args.seeds else None,
        split_name=str(args.split),
        top_k=int(args.top_k),
    )
    for result in results:
        print(f"model={result['model_name']}")
        print(f"seed={result['seed']}")
        print(f"supported={result['supported']}")
        print(f"output_root={result['output_root']}")
        print(f"explanation_count={result['explanation_count']}")
        print(f"top_scores_contact_sheet={result['top_scores_contact_sheet_path']}")
        print(f"anomalies_contact_sheet={result['anomalies_contact_sheet_path']}")
        print()


if __name__ == "__main__":
    main()
