import argparse
import random
import shutil
from pathlib import Path

def split_dataset(root: str, train_ratio: float = 0.8, seed: int = 42) -> None:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory '{root}' does not exist")

    graph_dirs = [p for p in root_path.iterdir() if p.is_dir()]
    if not graph_dirs:
        print("No graph directories found to split.")
        return

    random.seed(seed)
    random.shuffle(graph_dirs)
    split_idx = int(len(graph_dirs) * train_ratio)

    train_dir = root_path / "train"
    val_dir = root_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    for d in graph_dirs[:split_idx]:
        shutil.move(str(d), train_dir / d.name)
    for d in graph_dirs[split_idx:]:
        shutil.move(str(d), val_dir / d.name)

    print(f"Moved {split_idx} directories to {train_dir}")
    print(f"Moved {len(graph_dirs) - split_idx} directories to {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split graph directories into train/val subsets")
    parser.add_argument("root", help="root directory containing graph folders")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="proportion of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    split_dataset(args.root, args.train_ratio, args.seed)
