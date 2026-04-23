"""Accept synthetic records for benchmark when mask and review are present."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.synthetic import (  # noqa: E402
    accept_synthetic_records_for_benchmark,
    build_synthetic_pack_paths,
)
from towervision.utils.io import read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-name",
        default="anomaly_controlled_v1",
        help="Synthetic pack name under data/synthetic/<dataset>/<version>/.",
    )
    return parser.parse_args()


def main() -> None:
    """Mark curated synthetic records as accepted for benchmark."""

    args = parse_args()
    params = read_yaml(ROOT / "params.yaml")
    paths = build_synthetic_pack_paths(
        ROOT,
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        pack_name=str(args.pack_name),
    )
    summary = accept_synthetic_records_for_benchmark(paths)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
