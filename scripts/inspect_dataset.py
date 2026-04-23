"""Inspect the current dataset and generate technical documentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from towervision.data.inspect import (  # noqa: E402
    discover_default_dataset_path,
    inspect_dataset,
    render_terminal_summary,
    write_inspection_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Diretório ou arquivo .zip do dataset. Se omitido, tenta detectar automaticamente.",
    )
    return parser.parse_args()


def _resolve_repo_path(path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else ROOT / path_value


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = discover_default_dataset_path(ROOT)
    dataset_path = _resolve_repo_path(dataset_path)

    report = inspect_dataset(dataset_path)
    artifacts = write_inspection_artifacts(ROOT, report)

    print(render_terminal_summary(report))
    print(f"- timestamp: {artifacts.timestamp}")
    print(f"- markdown: {artifacts.markdown_path}")
    print(f"- csv: {artifacts.csv_path}")
    print(f"- latest: {artifacts.latest_markdown_path}")


if __name__ == "__main__":
    main()
