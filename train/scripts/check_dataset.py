"""Command line entry-point for dataset validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ngpathfinder.utils import data_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate dataset consistency across move tables, FASTQ entries and "
            "signal arrays."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help=(
            "Path to a dataset split (e.g. train/val/test) or a single segment "
            "directory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path: Path = args.dataset_path

    issues_found = False

    for segment_dir in data_validation.iter_segment_directories(dataset_path):
        segment_issues = data_validation.validate_segment(segment_dir)
        if segment_issues:
            issues_found = True
            for issue in segment_issues:
                print(issue.format())

    if not issues_found:
        print(f"All checks passed for dataset at {dataset_path}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())