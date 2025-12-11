"""Comprehensive dataset validation driven by an experiment config."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import ExperimentConfig, load_config
from ngpathfinder.utils import data_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all dataset splits referenced by an experiment config, "
            "including checks for NaN/Inf values in signal arrays."
        )
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=["train", "val", "test", "segment"],
        help=(
            "Optional subset of dataset entries to check. By default all available "
            "splits referenced by the config are validated."
        ),
    )
    return parser.parse_args()


def _resolve_dataset_paths(cfg: ExperimentConfig) -> Sequence[Tuple[str, Path]]:
    params = cfg.data.params
    splits: List[Tuple[str, Path]] = []

    def _resolve(path_str: str, split_name: str) -> None:
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        splits.append((split_name, path))

    train_dir = params.get("train_dir")
    val_dir = params.get("val_dir")
    test_dir = params.get("test_dir")

    if train_dir or val_dir or test_dir:
        if train_dir:
            _resolve(train_dir, "train")
        if val_dir:
            _resolve(val_dir, "val")
        if test_dir:
            _resolve(test_dir, "test")
        return splits

    segment_path = params.get("segment_path")
    if segment_path:
        _resolve(segment_path, "segment")
        return splits

    raise ValueError(
        "Experiment config does not define any dataset directories under data.params"
    )


def _numeric_issue(segment: Path, code: str, message: str) -> data_validation.ValidationIssue:
    return data_validation.ValidationIssue(
        segment=segment,
        read_id="<segment>",
        code=code,
        message=message,
    )


def _load_single_file(segment: Path, pattern: str, description: str) -> Path:
    matches = list(segment.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {description} ({pattern}) found in {segment}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {description} files found in {segment}: {matches}"
        )
    return matches[0]


def _check_numeric_health(segment: Path) -> List[data_validation.ValidationIssue]:
    issues: List[data_validation.ValidationIssue] = []

    try:
        mv_path = _load_single_file(segment, "*.mv.npy", "move table")
        mv_array = np.load(mv_path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - defensive against corrupt files
        issues.append(
            _numeric_issue(segment, "mv_load_error", f"Failed to load move table: {exc}")
        )
        mv_array = None

    try:
        signal_path = _load_single_file(segment, "*.signals.npy", "signal array")
        signal_array = np.load(signal_path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - defensive against corrupt files
        issues.append(
            _numeric_issue(segment, "signal_load_error", f"Failed to load signals: {exc}")
        )
        signal_array = None

    if mv_array is not None:
        if np.isnan(mv_array).any():
            issues.append(
                _numeric_issue(segment, "mv_nan", "Move table contains NaN values")
            )
        if np.isinf(mv_array).any():
            issues.append(
                _numeric_issue(segment, "mv_inf", "Move table contains infinite values")
            )

    if signal_array is not None:
        if np.isnan(signal_array).any():
            issues.append(
                _numeric_issue(segment, "signal_nan", "Signal array contains NaN values")
            )
        if np.isposinf(signal_array).any():
            issues.append(
                _numeric_issue(segment, "signal_posinf", "Signal array contains +Inf values")
            )
        if np.isneginf(signal_array).any():
            issues.append(
                _numeric_issue(segment, "signal_neginf", "Signal array contains -Inf values")
            )

    return issues


def _iter_segments(dataset_path: Path) -> Iterable[Path]:
    # Collect into a list to provide tqdm with a known length for progress reporting.
    segments = list(data_validation.iter_segment_directories(dataset_path))
    if not segments:
        raise ValueError(f"No segment directories found inside {dataset_path}")
    return segments


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    requested_splits = set(args.splits or [])
    dataset_entries = _resolve_dataset_paths(cfg)

    if requested_splits:
        dataset_entries = [
            (name, path) for name, path in dataset_entries if name in requested_splits
        ]
        if not dataset_entries:
            print(
                "No dataset entries match the provided --splits filters. Nothing to validate."
            )
            return 1

    total_segments = 0
    segments_with_issues = 0
    total_issues = 0

    for split_name, split_path in dataset_entries:
        print(f"Checking {split_name} dataset at {split_path}")

        try:
            segments = _iter_segments(split_path)
        except Exception as exc:
            print(f"  [error] {exc}")
            total_issues += 1
            continue

        total_segments += len(segments)
        issues_per_segment: Dict[Path, List[data_validation.ValidationIssue]] = defaultdict(list)

        progress = tqdm(segments, desc=f"{split_name} segments", unit="segment")
        for segment in progress:
            segment_issues: List[data_validation.ValidationIssue] = []
            try:
                segment_issues.extend(data_validation.validate_segment(segment))
            except Exception as exc:  # pragma: no cover - unexpected parsing errors
                segment_issues.append(
                    _numeric_issue(segment, "validation_error", str(exc))
                )

            segment_issues.extend(_check_numeric_health(segment))

            if segment_issues:
                issues_per_segment[segment].extend(segment_issues)

        progress.close()

        for segment, issues in sorted(issues_per_segment.items()):
            segments_with_issues += 1
            for issue in issues:
                print("  " + issue.format())
            total_issues += len(issues)

    if total_issues == 0:
        if total_segments == 0:
            print("No segments were found to validate.")
            return 1
        print(
            f"All checks passed across {total_segments} segments referenced in {args.config}."
        )
        return 0

    print(
        f"Found {total_issues} issues across {segments_with_issues} segments. "
        "See log above for details."
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())