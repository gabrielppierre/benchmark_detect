"""Render overlay previews for synthetic anomaly masks."""

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
    render_synthetic_overlay_contact_sheet,
    render_synthetic_mask_overlays,
)
from towervision.utils.io import ensure_dir, read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-name",
        default="anomaly_controlled_v1",
        help="Synthetic pack name under data/synthetic/<dataset>/<version>/.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Optional output directory for overlays. Defaults to "
            "reports/figures/<dataset>/<version>/<pack-name>/mask_overlays"
        ),
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns in the contact sheet.",
    )
    return parser.parse_args()


def main() -> None:
    """Render mask overlays for the active synthetic pack."""

    args = parse_args()
    params = read_yaml(ROOT / "params.yaml")
    dataset_name = str(params["dataset"]["name"])
    dataset_version = str(params["dataset"]["version"])
    paths = build_synthetic_pack_paths(
        ROOT,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        pack_name=str(args.pack_name),
    )
    output_root = (
        ROOT / str(args.output_dir)
        if args.output_dir
        else ROOT
        / "reports"
        / "figures"
        / dataset_name
        / dataset_version
        / str(args.pack_name)
        / "mask_overlays"
    )
    ensure_dir(output_root)
    rendered = render_synthetic_mask_overlays(paths, output_root=output_root)
    contact_sheet_summary = render_synthetic_overlay_contact_sheet(
        paths,
        overlay_root=output_root,
        output_path=output_root / "contact_sheet.png",
        columns=int(args.columns),
    )
    print(f"overlay_root={output_root}")
    print(f"overlay_count={len(rendered)}")
    print(f"contact_sheet={contact_sheet_summary['contact_sheet_path']}")


if __name__ == "__main__":
    main()
