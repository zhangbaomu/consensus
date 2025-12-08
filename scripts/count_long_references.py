"""Count segment reference FASTA lengths across dataset splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import ExperimentConfig, load_config  # noqa: E402
from ngpathfinder.utils.data_validation import (  # noqa: E402
    iter_segment_directories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an experiment config, resolve dataset splits, and count segment "
            "references whose FASTA length exceeds a specified threshold."
        )
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help=(
            "Maximum allowed reference length (in bases). Segments exceeding this value "
            "are counted."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=("train", "val", "test", "segment"),
        help="Optional subset of dataset entries to evaluate (defaults to all available).",
    )
    return parser.parse_args()


def _resolve_dataset_paths(cfg: ExperimentConfig) -> Sequence[Tuple[str, Path]]:
    params = cfg.data.params
    splits: List[Tuple[str, Path]] = []

    def _add(path_str: str | None, split_name: str) -> None:
        if not path_str:
            return
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        splits.append((split_name, path))

    _add(params.get("train_dir"), "train")
    _add(params.get("val_dir"), "val")
    _add(params.get("test_dir"), "test")

    if splits:
        return splits

    segment_path = params.get("segment_path")
    if segment_path:
        path = Path(segment_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return [("segment", path)]

    raise ValueError(
        "Experiment config does not define any dataset directories under data.params"
    )


def _iter_reference_fastas(segment_dir: Path) -> Iterable[Path]:
    patterns = ("*_reference.fasta", "*.fasta")
    for pattern in patterns:
        matches = sorted(segment_dir.glob(pattern))
        if matches:
            for match in matches:
                yield match
            return


def _compute_fasta_lengths(fasta_path: Path) -> List[int]:
    lengths: List[int] = []
    current_length = 0
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current_length:
                    lengths.append(current_length)
                    current_length = 0
                continue
            current_length += len(stripped)
    if current_length:
        lengths.append(current_length)
    return lengths


def main() -> int:
    args = parse_args()

    threshold = args.threshold
    if threshold <= 0:
        raise ValueError("--threshold must be a positive integer")

    cfg = load_config(args.config)
    dataset_entries = _resolve_dataset_paths(cfg)

    if args.splits:
        requested = set(args.splits)
        dataset_entries = [entry for entry in dataset_entries if entry[0] in requested]
        if not dataset_entries:
            print("No dataset entries match the provided --splits filters. Nothing to do.")
            return 1

    for split_name, split_path in dataset_entries:
        print(f"Split: {split_name}")
        print(f"  Dataset path: {split_path}")

        try:
            segments = list(iter_segment_directories(split_path))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"  [error] {exc}")
            continue

        if not segments:
            print("  [warning] No segment directories found.")
            continue

        exceeding = 0
        evaluated = 0
        missing_reference = 0

        for segment in segments:
            fasta_lengths: List[int] = []
            for fasta_path in _iter_reference_fastas(segment):
                fasta_lengths.extend(_compute_fasta_lengths(fasta_path))

            if not fasta_lengths:
                missing_reference += 1
                continue

            evaluated += 1
            if max(fasta_lengths) > threshold:
                exceeding += 1

        print(f"  Segments evaluated: {evaluated}")
        print(f"  Segments missing reference FASTA: {missing_reference}")
        print(f"  Segments exceeding threshold ({threshold} bp): {exceeding}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())