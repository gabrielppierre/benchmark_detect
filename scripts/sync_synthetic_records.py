"""Populate records.csv from generated synthetic outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.synthetic import build_synthetic_pack_paths, sync_records_from_generated_outputs  # noqa: E402
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
    """Synchronize records.csv using the current generated outputs."""

    args = parse_args()
    params = read_yaml(ROOT / "params.yaml")
    paths = build_synthetic_pack_paths(
        ROOT,
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        pack_name=str(args.pack_name),
    )
    records = sync_records_from_generated_outputs(paths)
    print(f"records_path={paths.records_path}")
    print(f"record_count={len(records)}")


if __name__ == "__main__":
    main()
