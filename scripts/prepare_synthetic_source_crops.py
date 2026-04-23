"""Export source crops for controlled synthetic anomaly generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.load import load_annotations, load_images_manifest, index_images_by_id  # noqa: E402
from towervision.data.synthetic import (  # noqa: E402
    export_synthetic_source_crops,
    initialize_controlled_synthetic_pack,
)
from towervision.utils.io import read_json, read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-name",
        default="anomaly_controlled_v1",
        help="Synthetic pack name under data/synthetic/<dataset>/<version>/.",
    )
    parser.add_argument(
        "--label",
        default="isoladores",
        help="Annotation label used to export source crops.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=64,
        help="Padding in pixels added around each GT bbox.",
    )
    parser.add_argument(
        "--shortlist-per-split",
        type=int,
        default=5,
        help="Number of recommended source crops per split in source_shortlist.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Export source crops from GT annotations in val/test."""

    args = parse_args()
    params = read_yaml(ROOT / "params.yaml")
    paths = params["paths"]
    pack_paths = initialize_controlled_synthetic_pack(
        ROOT,
        dataset_name=str(params["dataset"]["name"]),
        dataset_version=str(params["dataset"]["version"]),
        pack_name=str(args.pack_name),
        raw_dataset_root=ROOT / str(paths["raw_dataset_root"]),
    )
    images = load_images_manifest(ROOT / str(paths["cleaned_images_manifest"]))
    annotations = load_annotations(ROOT / str(paths["cleaned_annotations_manifest"]))
    split_mapping = read_json(ROOT / str(paths["splits_path"]))
    candidates, shortlist = export_synthetic_source_crops(
        pack_paths,
        index_images_by_id(images),
        annotations,
        split_mapping,
        label=str(args.label),
        padding=int(args.padding),
        shortlist_per_split=int(args.shortlist_per_split),
    )

    print(f"pack_root={pack_paths.root_dir}")
    print(f"source_candidates={pack_paths.source_candidates_path}")
    print(f"source_shortlist={pack_paths.source_shortlist_path}")
    for split_name, split_dir in pack_paths.source_crop_dirs.items():
        count = sum(1 for row in candidates if row["source_split"] == split_name)
        print(f"source_crops_{split_name}={split_dir} count={count}")
    print(f"shortlist_total={len(shortlist)}")


if __name__ == "__main__":
    main()
