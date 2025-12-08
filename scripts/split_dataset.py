"""Utility CLI for splitting segment directories into train/val subsets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(root: Path, train_ratio: float = 0.8, seed: int = 42) -> tuple[int, int]:
    """Split ``root`` directory into ``train`` and ``val`` subdirectories.

    Parameters
    ----------
    root:
        Root directory containing per-segment subdirectories to split.
    train_ratio:
        Proportion of segments to assign to the training split.
    seed:
        Random seed for shuffling segment order prior to splitting.

    Returns
    -------
    tuple[int, int]
        Counts of directories moved to the train and validation splits,
        respectively.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory '{root}' does not exist")

    graph_dirs = [p for p in root_path.iterdir() if p.is_dir()]
    if not graph_dirs:
        print("No graph directories found to split.")
        return 0, 0

    random.seed(seed)
    random.shuffle(graph_dirs)
    split_idx = int(len(graph_dirs) * train_ratio)

    train_dir = root_path / "train"
    val_dir = root_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    for directory in graph_dirs[:split_idx]:
        shutil.move(str(directory), train_dir / directory.name)
    for directory in graph_dirs[split_idx:]:
        shutil.move(str(directory), val_dir / directory.name)

    train_count = split_idx
    val_count = len(graph_dirs) - split_idx

    print(f"Moved {train_count} directories to {train_dir}")
    print(f"Moved {val_count} directories to {val_dir}")

    return train_count, val_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split graph directories into train/val subsets",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing per-segment graph folders.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of segments to assign to the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling shuffle order before splitting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dataset(args.root, args.train_ratio, args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())