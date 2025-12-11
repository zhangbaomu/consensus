"""Evaluate raw FASTQ read accuracy against reference sequences."""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import ExperimentConfig, load_config
from ngpathfinder.data import ReadDataset, list_block_segments
from ngpathfinder.data.datamodule import INV_BASE_VOCAB
from ngpathfinder.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw FASTQ reads against ground-truth references."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic read sampling within each segment",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional limit on the number of segments to evaluate",
    )
    return parser.parse_args()


def _dataset_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    use_fasta = params.get("use_fasta_reference", True)
    if not use_fasta:
        raise ValueError(
            "ReadDataset now requires FASTA references; set data.params.use_fasta_reference to true"
        )
    dataset_type = str(params.get("type", "legacy")).lower()
    return {
        "dataset_type": dataset_type,
        "max_mv_len": params.get("max_mv_len"),
        "max_reads_per_segment": params.get("max_reads_per_segment"),
        "fasta_glob_patterns": params.get("fasta_glob_patterns"),
        "ambiguous_base_policy": params.get("ambiguous_base_policy", "error"),
        "fasta_sequence_policy": params.get("fasta_sequence_policy", "first"),
        "suppress_mv_len_warnings": params.get("suppress_mv_len_warnings", False),
        "use_fastq_base_sequence": params.get("use_fastq_base_sequence", True),
        "use_read_flag": params.get("use_read_flag", True),
    }


def _resolve_path(path_str: str | None) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _split_segments(
    segment_names: Sequence[str], split_config: Mapping[str, Any], seed: int
) -> Tuple[List[str], List[str], List[str]]:
    if not segment_names:
        raise ValueError("No segments available to split")

    train_ratio = float(split_config.get("train", 0.8))
    val_ratio = float(split_config.get("val", 0.1))
    test_ratio = float(split_config.get("test", max(0.0, 1.0 - train_ratio - val_ratio)))

    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    if total_ratio > 1.0 + 1e-6:
        raise ValueError("Split ratios may not sum to more than 1")

    names = list(segment_names)
    names.sort()
    rng = random.Random(seed)
    rng.shuffle(names)

    normalized = [train_ratio, val_ratio, test_ratio]
    ratio_sum = sum(normalized)
    normalized = [value / ratio_sum for value in normalized]

    counts = [int(len(names) * value) for value in normalized]
    while sum(counts) < len(names):
        for idx in range(len(counts)):
            if sum(counts) >= len(names):
                break
            counts[idx] += 1

    train_count, val_count, test_count = counts
    if train_count <= 0:
        raise ValueError("Train split must contain at least one segment")

    train = names[:train_count]
    val = names[train_count : train_count + val_count]
    test = names[train_count + val_count : train_count + val_count + test_count]
    return train, val, test


def _resolve_dataset(cfg: ExperimentConfig, split: str) -> ReadDataset:
    dataset_name = cfg.data.dataset.lower()
    if dataset_name not in {"", "read", "readdataset", "read_dataset"}:
        raise ValueError(f"Unsupported dataset type '{cfg.data.dataset}' for raw read evaluation")

    params = cfg.data.params
    dataset_params = _dataset_kwargs(params)
    dataset_type = dataset_params.get("dataset_type", "legacy")

    if dataset_type in {"legacy", "npy"}:
        key = f"{split}_dir"
        path = _resolve_path(params.get(key))
        if not path:
            raise ValueError(
                f"Config is missing data.params.{key!s}; cannot construct '{split}' dataset for evaluation"
            )
        return ReadDataset(path, **dataset_params)

    dataset_root = _resolve_path(params.get("dataset_dir"))
    if dataset_root is None:
        raise ValueError("Monolithic datasets require 'dataset_dir' under data.params")

    explicit_segment = params.get("segment")
    if explicit_segment:
        return ReadDataset(dataset_root, segment_names=[explicit_segment], **dataset_params)

    split_config = params.get("split", {}) or {}
    split_seed = int(params.get("split_seed", cfg.seed))
    available_segments = list_block_segments(dataset_root)
    train_segments, val_segments, test_segments = _split_segments(
        available_segments, split_config, split_seed
    )

    selected: List[str]
    if split == "train":
        selected = train_segments
    elif split == "val":
        selected = val_segments
    else:
        selected = test_segments

    if not selected:
        raise ValueError(f"No segments available for split '{split}'")

    return ReadDataset(dataset_root, segment_names=selected, **dataset_params)


def _compute_alignment_stats(reference: str, hypothesis: str) -> Tuple[int, int, int, int]:
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if ref_len == 0 and hyp_len == 0:
        return 0, 0, 0, 0

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    back: List[List[str]] = [["" for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, ref_len + 1):
        ref_char = reference[i - 1]
        for j in range(1, hyp_len + 1):
            hyp_char = hypothesis[j - 1]
            substitution_cost = dp[i - 1][j - 1] + (0 if ref_char == hyp_char else 1)
            deletion_cost = dp[i - 1][j] + 1
            insertion_cost = dp[i][j - 1] + 1

            best_cost = substitution_cost
            op = "M" if ref_char == hyp_char else "S"
            if deletion_cost < best_cost:
                best_cost = deletion_cost
                op = "D"
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                op = "I"

            dp[i][j] = best_cost
            back[i][j] = op

    distance = dp[ref_len][hyp_len]
    matches = 0
    alignment_length = 0

    i, j = ref_len, hyp_len
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"M", "S"}:
            if op == "M":
                matches += 1
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            j -= 1
        else:
            break
        alignment_length += 1

    return distance, alignment_length, matches, ref_len


def _recover_read_sequence(moves, base_indices) -> str:
    if moves.shape[0] != base_indices.shape[0]:
        raise ValueError("Move tensor and base index tensor must have matching lengths")

    bases: List[str] = []
    for move_value, base_value in zip(moves.tolist(), base_indices.tolist()):
        if move_value > 0.5:
            base = INV_BASE_VOCAB.get(int(base_value), "N")
            bases.append(base)
    return "".join(bases)


def _select_read_index(num_reads: int, rng: random.Random) -> int:
    if num_reads <= 0:
        raise ValueError("Segments must contain at least one read")
    if num_reads == 1:
        return 0
    return rng.randrange(num_reads)


def _evaluate_raw_reads(
    dataset: ReadDataset,
    *,
    max_segments: Optional[int],
    rng: random.Random,
    logger: logging.Logger,
) -> None:
    total_matches = 0
    total_alignment_length = 0
    total_ref_length = 0
    acc_aligned_list: List[float] = []
    acc_ref_list: List[float] = []
    distance_list: List[float] = []
    normalized_distance_list: List[float] = []

    evaluated_segments = 0
    zero_length_reads = 0
    examples: List[Tuple[str, str, int, int, float, float]] = []

    num_segments = len(dataset)
    if max_segments is not None:
        num_segments = min(num_segments, max_segments)

    logger.info("Evaluating %d segment(s) for raw read accuracy", num_segments)

    for idx in tqdm(range(num_segments), desc="Raw read evaluation", unit="segment"):
        sample = dataset[idx]
        read_count = len(sample["read_id"])
        if read_count == 0:
            logger.warning("Segment %s does not contain any reads; skipping", sample["segment_id"])
            continue

        chosen = _select_read_index(read_count, rng)
        moves = sample["move"][chosen]
        base_indices = sample["base_index"][chosen]

        read_sequence = _recover_read_sequence(moves, base_indices)
        reference_sequence = str(sample["reference_string"])

        distance, alignment_len, matches, ref_len = _compute_alignment_stats(
            reference_sequence, read_sequence
        )

        if alignment_len == 0:
            acc_aligned = 1.0
        else:
            acc_aligned = matches / float(alignment_len)
        acc_ref = matches / float(ref_len) if ref_len > 0 else 1.0
        normalized_distance = distance / float(ref_len) if ref_len > 0 else 0.0

        acc_aligned_list.append(acc_aligned)
        acc_ref_list.append(acc_ref)
        distance_list.append(float(distance))
        normalized_distance_list.append(normalized_distance)

        total_matches += matches
        total_alignment_length += alignment_len
        total_ref_length += ref_len
        evaluated_segments += 1

        if len(read_sequence) == 0:
            zero_length_reads += 1

        segment_id = str(sample["segment_id"])
        read_id = str(sample["read_id"][chosen])

        logger.debug(
            "Segment %s | read=%s | length=%d | distance=%d | acc_aligned=%.4f | acc_ref=%.4f",
            segment_id,
            read_id,
            len(read_sequence),
            distance,
            acc_aligned,
            acc_ref,
        )

        if len(examples) < 10:
            examples.append(
                (
                    segment_id,
                    read_id,
                    len(read_sequence),
                    distance,
                    acc_aligned,
                    acc_ref,
                )
            )

    if evaluated_segments == 0:
        logger.info("No segments were evaluated; aborting metrics computation")
        return

    mean_acc_aligned = sum(acc_aligned_list) / evaluated_segments
    mean_acc_ref = sum(acc_ref_list) / evaluated_segments
    mean_distance = sum(distance_list) / evaluated_segments
    mean_normalized_distance = (
        sum(normalized_distance_list) / evaluated_segments if normalized_distance_list else 0.0
    )

    overall_acc_aligned = (
        total_matches / total_alignment_length if total_alignment_length > 0 else 1.0
    )
    overall_acc_ref = total_matches / total_ref_length if total_ref_length > 0 else 1.0

    logger.info("Raw read evaluation complete for %d segment(s)", evaluated_segments)
    logger.info("  Mean Acc@aligned: %.4f", mean_acc_aligned)
    logger.info("  Overall base-level accuracy (alignment-aware): %.4f", overall_acc_aligned)
    logger.info("  Mean Acc@pos (reference): %.4f", mean_acc_ref)
    logger.info("  Overall Acc@pos (reference): %.4f", overall_acc_ref)
    logger.info("  Mean Levenshtein distance: %.4f", mean_distance)
    logger.info("  Mean normalized Levenshtein distance: %.4f", mean_normalized_distance)
    logger.info(
        "  Zero-length raw reads: %d (%.2f%%)",
        zero_length_reads,
        (zero_length_reads / evaluated_segments) * 100.0,
    )

    if examples:
        logger.info("Sample raw read alignments (up to 10 records):")
        for segment_id, read_id, length, distance, acc_aligned, acc_ref in examples:
            logger.info(
                "  Segment %s | read=%s | length=%d | distance=%d | acc_aligned=%.4f | acc_ref=%.4f",
                segment_id,
                read_id,
                length,
                distance,
                acc_aligned,
                acc_ref,
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    configure_logging(cfg.output_dir)
    logger = logging.getLogger("eval_raw_read")

    rng = random.Random(args.seed)

    max_segments = args.max_segments
    if max_segments is not None and max_segments <= 0:
        raise ValueError("--max-segments must be a positive integer when provided")

    dataset = _resolve_dataset(cfg, args.split)
    _evaluate_raw_reads(
        dataset,
        max_segments=max_segments,
        rng=rng,
        logger=logger,
    )


if __name__ == "__main__":
    main()