"""Initialize a controlled synthetic anomaly pack for the active dataset version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.synthetic import initialize_controlled_synthetic_pack  # noqa: E402
from towervision.utils.io import read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-name",
        default="anomaly_controlled_v1",
        help="Synthetic pack name under data/synthetic/<dataset>/<version>/.",
    )
    parser.add_argument(
        "--images-per-generator",
        type=int,
        default=5,
        help="Planned number of accepted images per generator.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the controlled synthetic pack for the active dataset version."""

    args = parse_args()
    params = read_yaml(ROOT / "params.yaml")
    paths = initialize_controlled_synthetic_pack(
        ROOT,
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        pack_name=str(args.pack_name),
        raw_dataset_root=ROOT / str(params["paths"]["raw_dataset_root"]),
        images_per_generator=int(args.images_per_generator),
    )
    print(f"root={paths.root_dir}")
    print(f"manifest={paths.manifest_path}")
    print(f"records={paths.records_path}")
    print(f"readme={paths.readme_path}")


if __name__ == "__main__":
    main()
