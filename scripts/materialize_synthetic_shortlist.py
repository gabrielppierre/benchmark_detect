"""Copy the current synthetic source shortlist into one handoff folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.synthetic import (  # noqa: E402
    build_synthetic_pack_paths,
    materialize_shortlist_bundle,
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
    """Copy shortlist crops into a single handoff directory."""

    args = parse_args()
    params = read_yaml(ROOT / "params.yaml")
    paths = build_synthetic_pack_paths(
        ROOT,
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        pack_name=str(args.pack_name),
    )
    copied_rows = materialize_shortlist_bundle(paths)
    print(f"bundle_dir={paths.source_shortlist_bundle_dir}")
    print(f"bundle_count={len(copied_rows)}")


if __name__ == "__main__":
    main()
