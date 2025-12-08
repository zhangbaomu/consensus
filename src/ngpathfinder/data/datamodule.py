"""Dataset orchestration utilities."""
from __future__ import annotations

import csv
import functools
import gzip
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4}
INV_BASE_VOCAB = {value: key for key, value in BASE_VOCAB.items()}

DEFAULT_FASTA_PATTERNS: Tuple[str, ...] = (
    "*_reference.fasta",
    "*.fasta",
    "*.fa",
    "*.fasta.gz",
    "*.fa.gz",
)

AMBIGUOUS_BASE_POLICIES = {"error", "skip"}
FASTA_SEQUENCE_POLICIES = {"first", "longest"}


@dataclass(frozen=True)
class _CollateTransformOptions:
    robust_normalization: bool = False
    robust_eps: float = 1e-3
    time_mask_enabled: bool = False
    time_mask_min_masks: int = 0
    time_mask_max_masks: int = 0
    time_mask_min_width: int = 0
    time_mask_max_width: int = 0
    time_mask_probability: float = 1.0
    time_stretch_enabled: bool = False
    time_stretch_probability: float = 0.0
    time_stretch_min_scale: float = 1.0
    time_stretch_max_scale: float = 1.0
    additive_noise_enabled: bool = False
    additive_noise_probability: float = 0.0
    additive_noise_std_min: float = 0.0
    additive_noise_std_max: float = 0.0


_DEFAULT_COLLATE_OPTIONS = _CollateTransformOptions()


def _options_is_noop(options: _CollateTransformOptions) -> bool:
    return not (
        options.robust_normalization
        or options.time_mask_enabled
        or options.time_stretch_enabled
        or options.additive_noise_enabled
    )


def _parse_probability(value: Any, name: str, default: float = 1.0) -> float:
    if value is None:
        probability = float(default)
    else:
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a float between 0 and 1") from exc
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must lie within [0, 1]")
    return probability


def _parse_int_range(value: Any, name: str, *, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        lower, upper = default
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        try:
            lower = upper = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an int or a length-2 sequence of ints") from exc
    lower = int(lower)
    upper = int(upper)
    if lower < 0 or upper < 0:
        raise ValueError(f"{name} must not be negative")
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return lower, upper


def _parse_float_range(value: Any, name: str, *, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        lower, upper = default
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        try:
            lower = upper = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a float or a length-2 sequence of floats") from exc
    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    if lower <= 0 or upper <= 0:
        raise ValueError(f"{name} values must be strictly positive")
    return lower, upper


def _parse_collate_transform_config(
    raw: Mapping[str, Any] | _CollateTransformOptions | None,
) -> _CollateTransformOptions:
    if raw is None:
        return _DEFAULT_COLLATE_OPTIONS
    if isinstance(raw, _CollateTransformOptions):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("Collate transform configuration must be a mapping")

    robust_cfg = raw.get("robust_normalization") or raw.get("robust_norm")
    robust_enabled = False
    robust_eps = 1e-3
    if isinstance(robust_cfg, Mapping):
        robust_enabled = bool(robust_cfg.get("enabled", robust_cfg.get("enable", False)))
        robust_eps = float(robust_cfg.get("eps", robust_cfg.get("epsilon", 1e-3)))
    elif isinstance(robust_cfg, bool):
        robust_enabled = robust_cfg
    elif robust_cfg is not None:
        raise TypeError("robust_normalization must be configured with a mapping or boolean")
    if robust_eps <= 0:
        raise ValueError("robust_normalization.eps must be > 0")

    augment_cfg = raw.get("augmentations") or raw.get("augmentation") or {}
    if augment_cfg is None:
        augment_cfg = {}
    if augment_cfg and not isinstance(augment_cfg, Mapping):
        raise TypeError("augmentations must be a mapping when provided")
    augment_enabled = bool(augment_cfg.get("enabled", True))

    time_mask_cfg = augment_cfg.get("time_mask") if isinstance(augment_cfg, Mapping) else None
    time_mask_enabled = bool(
        augment_enabled
        and isinstance(time_mask_cfg, Mapping)
        and time_mask_cfg.get("enabled", True)
    )
    time_mask_min_masks = 0
    time_mask_max_masks = 0
    time_mask_min_width = 0
    time_mask_max_width = 0
    time_mask_probability = 1.0
    if time_mask_enabled:
        time_mask_min_masks, time_mask_max_masks = _parse_int_range(
            time_mask_cfg.get("count", time_mask_cfg.get("num_masks")),
            "augmentations.time_mask.count",
            default=(1, 2),
        )
        time_mask_min_width, time_mask_max_width = _parse_int_range(
            time_mask_cfg.get("width"),
            "augmentations.time_mask.width",
            default=(5, 20),
        )
        time_mask_probability = _parse_probability(
            time_mask_cfg.get("probability"),
            "augmentations.time_mask.probability",
        )
        if time_mask_min_masks == 0 or time_mask_min_width == 0:
            time_mask_enabled = False

    time_stretch_cfg = augment_cfg.get("time_stretch") if isinstance(augment_cfg, Mapping) else None
    time_stretch_enabled = bool(
        augment_enabled
        and isinstance(time_stretch_cfg, Mapping)
        and time_stretch_cfg.get("enabled", True)
    )
    time_stretch_probability = 0.0
    time_stretch_min_scale = 1.0
    time_stretch_max_scale = 1.0
    if time_stretch_enabled:
        time_stretch_probability = _parse_probability(
            time_stretch_cfg.get("probability"),
            "augmentations.time_stretch.probability",
            default=1.0,
        )
        scale_range = time_stretch_cfg.get("scale") or time_stretch_cfg.get("range")
        if scale_range is not None:
            time_stretch_min_scale, time_stretch_max_scale = _parse_float_range(
                scale_range,
                "augmentations.time_stretch.scale",
                default=(0.97, 1.03),
            )
        else:
            max_delta = float(time_stretch_cfg.get("max_scale", 0.0))
            if max_delta < 0:
                raise ValueError("augmentations.time_stretch.max_scale must be >= 0")
            time_stretch_min_scale = max(0.0, 1.0 - max_delta)
            time_stretch_max_scale = 1.0 + max_delta
        if time_stretch_probability == 0.0 or abs(time_stretch_max_scale - 1.0) < 1e-6:
            time_stretch_enabled = False

    noise_cfg = augment_cfg.get("additive_noise") if isinstance(augment_cfg, Mapping) else None
    additive_noise_enabled = bool(
        augment_enabled and isinstance(noise_cfg, Mapping) and noise_cfg.get("enabled", True)
    )
    additive_noise_probability = 0.0
    additive_noise_std_min = 0.0
    additive_noise_std_max = 0.0
    if additive_noise_enabled:
        additive_noise_probability = _parse_probability(
            noise_cfg.get("probability"),
            "augmentations.additive_noise.probability",
            default=1.0,
        )
        std_range = noise_cfg.get("std") or noise_cfg.get("sigma")
        additive_noise_std_min, additive_noise_std_max = _parse_float_range(
            std_range,
            "augmentations.additive_noise.std",
            default=(0.01, 0.03),
        )
        if additive_noise_probability == 0.0:
            additive_noise_enabled = False

    return _CollateTransformOptions(
        robust_normalization=robust_enabled,
        robust_eps=robust_eps,
        time_mask_enabled=time_mask_enabled,
        time_mask_min_masks=time_mask_min_masks,
        time_mask_max_masks=time_mask_max_masks,
        time_mask_min_width=time_mask_min_width,
        time_mask_max_width=time_mask_max_width,
        time_mask_probability=time_mask_probability,
        time_stretch_enabled=time_stretch_enabled,
        time_stretch_probability=time_stretch_probability,
        time_stretch_min_scale=time_stretch_min_scale,
        time_stretch_max_scale=time_stretch_max_scale,
        additive_noise_enabled=additive_noise_enabled,
        additive_noise_probability=additive_noise_probability,
        additive_noise_std_min=additive_noise_std_min,
        additive_noise_std_max=additive_noise_std_max,
    )


@dataclass(frozen=True)
class _FastaReferenceOptions:
    glob_patterns: Tuple[str, ...] = DEFAULT_FASTA_PATTERNS
    ambiguous_base_policy: str = "error"
    fasta_sequence_policy: str = "first"


def _open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    sequence_parts: List[str] = []

    with _open_text_file(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if header is not None:
        records.append((header, "".join(sequence_parts)))

    if not records:
        raise ValueError(f"FASTA file '{path}' did not contain any sequences")

    return records


def _select_fasta_sequence(
    records: Sequence[Tuple[str, str]],
    policy: str,
) -> Tuple[str, str]:
    if policy not in FASTA_SEQUENCE_POLICIES:
        raise ValueError(f"Unsupported fasta_sequence_policy='{policy}'")

    if policy == "first":
        return records[0]

    # policy == "longest"
    longest = max(records, key=lambda item: len(item[1]))
    return longest


def _sanitize_reference_sequence(
    sequence: str,
    *,
    path: Path,
    policy: str,
) -> str:
    cleaned: List[str] = []
    for index, base in enumerate(sequence):
        normalized = base.upper()
        if normalized in BASE_VOCAB:
            cleaned.append(normalized)
            continue
        if policy == "skip":
            continue
        raise ValueError(
            f"Encountered unsupported base '{base}' at position {index} in '{path}'"
        )

    sanitized = "".join(cleaned)
    if not sanitized:
        raise ValueError(f"Reference sequence from '{path}' is empty after sanitisation")
    return sanitized


def _find_fasta_reference(
    segment_dir: Path,
    options: _FastaReferenceOptions,
) -> Tuple[torch.Tensor, str, Path]:
    for pattern in options.glob_patterns:
        matches = sorted(segment_dir.glob(pattern))
        if not matches:
            continue
        if len(matches) > 1:
            joined = ", ".join(match.name for match in matches)
            raise ValueError(
                f"Multiple FASTA files matched pattern '{pattern}' in '{segment_dir}': {joined}"
            )

        fasta_path = matches[0]
        records = _parse_fasta(fasta_path)
        _, sequence = _select_fasta_sequence(records, options.fasta_sequence_policy)
        sanitized = _sanitize_reference_sequence(
            sequence,
            path=fasta_path,
            policy=options.ambiguous_base_policy,
        )
        indices = torch.tensor([BASE_VOCAB[base] for base in sanitized], dtype=torch.long)
        return indices, sanitized, fasta_path

    raise FileNotFoundError(
        f"No FASTA reference found in '{segment_dir}'. Checked patterns: {options.glob_patterns}"
    )


def _read_fastq_sequences(
    path: Path, allowed_read_ids: Optional[Set[str]] = None
) -> Dict[str, str]:
    """Parse a FASTQ file and return read id to base sequence mapping.

    Parameters
    ----------
    path:
        FASTQ file containing the reads.
    allowed_read_ids:
        Optional set of read identifiers to keep. When provided, sequences
        whose identifiers are not in the set will be skipped to reduce memory
        pressure when working with monolithic datasets.
    """

    sequences: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if not header.startswith("@"):
                raise ValueError(f"Malformed FASTQ header in {path}: {header!r}")

            read_id = header[1:].strip()

            sequence_line = handle.readline()
            if not sequence_line:
                raise ValueError(
                    f"FASTQ file {path} is truncated: sequence line missing for read '{read_id}'"
                )
            sequence = sequence_line.strip()

            separator_line = handle.readline()
            if not separator_line:
                raise ValueError(
                    f"FASTQ file {path} is truncated: separator line missing for read '{read_id}'"
                )
            separator = separator_line.strip()
            if separator and separator != "+":
                raise ValueError(
                    f"FASTQ file {path} has invalid separator for read '{read_id}': {separator!r}"
                )

            quality_line = handle.readline()
            if quality_line == "":
                raise ValueError(
                    f"FASTQ file {path} is truncated: quality line missing for read '{read_id}'"
                )
            # Some datasets omit quality values but still include the newline. Accept it.

            if allowed_read_ids is None or read_id in allowed_read_ids:
                sequences[read_id] = sequence
    return sequences


def _read_fastq_sequences_multimap(
    path: Path, allowed_read_ids: Optional[Set[str]] = None
) -> Dict[str, List[str]]:
    """
    Parse a FASTQ file and return mapping from read id to all base sequences (in file order).

    This variant keeps *all* occurrences for a read_id (not just the last), which is needed
    for block/monolithic datasets where identical read_ids can appear multiple times.
    """

    sequences: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if not header.startswith("@"):
                raise ValueError(f"Malformed FASTQ header in {path}: {header!r}")

            read_id = header[1:].strip()

            sequence_line = handle.readline()
            if not sequence_line:
                raise ValueError(
                    f"FASTQ file {path} is truncated: sequence line missing for read '{read_id}'"
                )
            sequence = sequence_line.strip()

            separator_line = handle.readline()
            if not separator_line:
                raise ValueError(
                    f"FASTQ file {path} is truncated: separator line missing for read '{read_id}'"
                )
            separator = separator_line.strip()
            if separator and separator != "+":
                raise ValueError(
                    f"FASTQ file {path} has invalid separator for read '{read_id}': {separator!r}"
                )

            quality_line = handle.readline()
            if quality_line == "":
                raise ValueError(
                    f"FASTQ file {path} is truncated: quality line missing for read '{read_id}'"
                )

            if allowed_read_ids is None or read_id in allowed_read_ids:
                sequences.setdefault(read_id, []).append(sequence)
    return sequences


def _expand_base_sequence(sequence: str, moves: Sequence[int]) -> torch.Tensor:
    """Expand a compact base sequence using the move table into index tensors.

    Move values represent how many bases to advance before emitting the base
    associated with a timestep. This means the total number of bases consumed
    should equal ``sum(moves)`` (rather than simply counting non-zero entries),
    and values greater than 1 are treated as multi-base hops.
    """

    move_list = list(moves)
    expanded = torch.zeros(len(move_list), dtype=torch.long)
    consumed = 0
    total_bases = len(sequence)

    for idx, mv in enumerate(move_list):
        if mv:
            if mv < 0:
                raise ValueError("Move values must be non-negative")
            consumed += mv
            if consumed > total_bases:
                raise ValueError("Move table expects more bases than provided")
            base = sequence[consumed - 1]
            expanded[idx] = BASE_VOCAB.get(base.upper(), 0)

    if consumed != total_bases:
        raise ValueError(
            "Base sequence contains more symbols than moves allow (sum of moves is too small)"
        )

    return expanded


def _apply_time_mask(signal: torch.Tensor, options: _CollateTransformOptions) -> None:
    if not options.time_mask_enabled:
        return
    if random.random() > options.time_mask_probability:
        return
    length = signal.size(0)
    if length <= 0:
        return
    max_masks = min(options.time_mask_max_masks, length)
    min_masks = min(options.time_mask_min_masks, max_masks)
    if max_masks <= 0:
        return
    num_masks = random.randint(min_masks, max_masks)
    if num_masks <= 0:
        return
    max_width = min(options.time_mask_max_width, length)
    min_width = min(options.time_mask_min_width, max_width)
    if max_width <= 0:
        return
    for _ in range(num_masks):
        width = random.randint(min_width, max_width)
        width = max(1, min(width, length))
        if width >= length:
            start = 0
            end = length
        else:
            start = random.randint(0, length - width)
            end = start + width
        signal[start:end].zero_()


def _apply_time_stretch(signal: torch.Tensor, options: _CollateTransformOptions) -> None:
    if not options.time_stretch_enabled:
        return
    if random.random() > options.time_stretch_probability:
        return
    length = signal.size(0)
    if length <= 1:
        return
    scale = random.uniform(options.time_stretch_min_scale, options.time_stretch_max_scale)
    if abs(scale - 1.0) < 1e-3:
        return
    channels = signal.size(1)
    src = signal.transpose(0, 1).unsqueeze(0)
    intermediate = max(2, int(round(length * scale)))
    if intermediate == length:
        return
    resampled = F.interpolate(src, size=intermediate, mode="linear", align_corners=False)
    resampled = F.interpolate(resampled, size=length, mode="linear", align_corners=False)
    signal.copy_(resampled.squeeze(0).transpose(0, 1))


def _apply_additive_noise(signal: torch.Tensor, options: _CollateTransformOptions) -> None:
    if not options.additive_noise_enabled:
        return
    if random.random() > options.additive_noise_probability:
        return
    std = random.uniform(options.additive_noise_std_min, options.additive_noise_std_max)
    if std <= 0:
        return
    noise = torch.randn_like(signal) * std
    signal.add_(noise)


def _apply_collate_transforms(
    signals: torch.Tensor,
    lengths: torch.Tensor,
    read_padding_mask: torch.Tensor,
    options: _CollateTransformOptions,
) -> None:
    if _options_is_noop(options):
        return

    batch_size, max_reads, _, _ = signals.shape
    for batch_idx in range(batch_size):
        for read_idx in range(max_reads):
            if read_padding_mask[batch_idx, read_idx]:
                continue
            length = int(lengths[batch_idx, read_idx])
            if length <= 0:
                continue
            signal = signals[batch_idx, read_idx, :length]
            if options.robust_normalization:
                median = signal.median(dim=0, keepdim=True).values
                deviation = (signal - median).abs().median(dim=0, keepdim=True).values
                deviation = deviation.clamp_min(options.robust_eps)
                signal.sub_(median).div_(deviation)
            _apply_time_mask(signal, options)
            _apply_time_stretch(signal, options)
            _apply_additive_noise(signal, options)
@dataclass
class _SegmentCache:
    """Lightweight container holding paths and metadata for one segment directory."""

    path: Path
    prefix: str
    signals_path: Path
    moves_path: Path
    sequences: Dict[str, str]
    entries: List[Dict[str, Any]]
    reference_index: Optional[torch.Tensor]
    reference_string: Optional[str]
    reference_path: Optional[Path]
    _signals_memmap: Optional[np.memmap] = None
    _moves_memmap: Optional[np.memmap] = None
    _signals_loader: Optional[Callable[[], np.ndarray]] = None
    _moves_loader: Optional[Callable[[], np.ndarray]] = None

    def signals_memmap(self) -> np.memmap:
        if self._signals_memmap is None:
            if self._signals_loader is not None:
                self._signals_memmap = self._signals_loader()
            else:
                self._signals_memmap = np.load(self.signals_path, mmap_mode="r")
        return self._signals_memmap

    def moves_memmap(self) -> np.memmap:
        if self._moves_memmap is None:
            if self._moves_loader is not None:
                self._moves_memmap = self._moves_loader()
            else:
                self._moves_memmap = np.load(self.moves_path, mmap_mode="r")
        return self._moves_memmap


def _is_segment_directory(path: Path) -> bool:
    """Return True if ``path`` looks like a single read segment directory."""

    if not path.is_dir():
        return False
    prefix = path.name
    required = (
        path / f"{prefix}.index.tsv",
        path / f"{prefix}.reads.fastq",
        path / f"{prefix}.mv.npy",
        path / f"{prefix}.signals.npy",
    )
    return all(p.exists() for p in required)


def _read_index(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entries.append(
                {
                    "read_id": row["read_id"],
                    "offset": int(row["offset"]),
                    "length": int(row["length"]),
                    "mv_offset": int(row["mv_offset"]),
                    "mv_len": int(row["mv_len"]),
                    "stride": int(row["stride"]),
                    "flag": int(row["flag"]),
                }
            )
    return entries


def _load_segment(segment_dir: Path, *, fasta_options: _FastaReferenceOptions) -> _SegmentCache:
    prefix = segment_dir.name
    index_path = segment_dir / f"{prefix}.index.tsv"
    fastq_path = segment_dir / f"{prefix}.reads.fastq"
    mv_path = segment_dir / f"{prefix}.mv.npy"
    signal_path = segment_dir / f"{prefix}.signals.npy"

    for path in (index_path, fastq_path, mv_path, signal_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset component missing: {path}")

    sequences = _read_fastq_sequences(fastq_path)
    entries = _read_index(index_path)

    reference_index, reference_string, reference_path = _find_fasta_reference(
        segment_dir, fasta_options
    )

    return _SegmentCache(
        path=segment_dir,
        prefix=prefix,
        signals_path=signal_path,
        moves_path=mv_path,
        sequences=sequences,
        entries=entries,
        reference_index=reference_index,
        reference_string=reference_string,
        reference_path=reference_path,
    )


def _lazy_memmap(path: Path, dtype: np.dtype) -> Callable[[], np.ndarray]:
    memmap: Dict[str, np.ndarray] = {}

    def _loader() -> np.ndarray:
        if "array" not in memmap:
            memmap["array"] = np.memmap(path, dtype=dtype, mode="r")
        return memmap["array"]

    return _loader


def list_block_segments(dataset_root: Path) -> List[str]:
    """Return sorted region identifiers from ``region_blocks.tsv``.

    The function only reads lightweight metadata, making it suitable for
    planning deterministic train/val/test splits without loading the full
    dataset into memory.
    """

    blocks_path = dataset_root / "region_blocks.tsv"
    if not blocks_path.exists():
        raise FileNotFoundError(f"Missing region_blocks.tsv in {dataset_root}")

    regions: List[str] = []
    with blocks_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = reader.fieldnames or []
        if "region" not in headers:
            raise ValueError(
                f"region_blocks.tsv in {dataset_root} must contain a 'region' column"
            )
        for row in reader:
            region = row.get("region")
            if region:
                regions.append(region)

    unique = sorted(dict.fromkeys(regions))
    if not unique:
        raise ValueError(f"No regions listed in {blocks_path}")
    return unique


def _read_block_index(
    index_path: Path, allowed_regions: Optional[Set[str]] = None
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    ordered: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = reader.fieldnames or []
        required = {"region", "read_id"}
        missing = required.difference(headers)
        if missing:
            raise ValueError(
                f"Index file {index_path} is missing required columns: {sorted(missing)}"
            )

        def _get_int(row: Dict[str, str], key: str, fallback: Optional[str] = None) -> int:
            value = row.get(key)
            if value is None and fallback is not None:
                value = row.get(fallback)
            if value is None:
                raise KeyError(
                    f"Index file {index_path} is missing required field '{key}'"
                )
            return int(value)

        for row in reader:
            region = row["region"]
            if allowed_regions is not None and region not in allowed_regions:
                continue
            entry = {
                "read_id": row["read_id"],
                "offset": _get_int(row, "sig_offset", "offset"),
                "length": _get_int(row, "sig_len", "length"),
                "mv_offset": _get_int(row, "mv_offset"),
                "mv_len": _get_int(row, "mv_len"),
                "stride": _get_int(row, "stride", "stride"),
                "flag": int(row.get("flag", 0)),
            }
            grouped.setdefault(region, []).append(entry)
            ordered.append(entry)

    return grouped, ordered


def _validate_block_moves(
    *,
    entries_by_region: Dict[str, List[Dict[str, Any]]],
    moves_loader: Callable[[], np.ndarray],
    dataset_root: Path,
) -> None:
    """Ensure each read's sequence length does not exceed the summed moves."""

    moves = moves_loader()
    total_moves = moves.shape[0]

    for region, entries in entries_by_region.items():
        for entry in entries:
            read_id = entry["read_id"]
            mv_offset = entry["mv_offset"]
            mv_len = entry["mv_len"]
            mv_end = mv_offset + mv_len

            if mv_offset < 0 or mv_end > total_moves:
                raise ValueError(
                    (
                        "Move slice is out of bounds for read '{read_id}' in region '{region}' "
                        "(mv_offset={mv_offset}, mv_len={mv_len}, total_moves={total_moves}) "
                        "under dataset {root}"
                    ).format(
                        read_id=read_id,
                        region=region,
                        mv_offset=mv_offset,
                        mv_len=mv_len,
                        total_moves=total_moves,
                        root=dataset_root,
                    )
                )

            move_slice = moves[mv_offset:mv_end]
            move_steps = int(np.sum(move_slice, dtype=np.int64))
            non_zero = int(np.count_nonzero(move_slice))
            sequence = entry.get("sequence", "")
            seq_len = len(sequence)

            if seq_len > move_steps:
                raise ValueError(
                    "Move/base mismatch in dataset {root}: region='{region}', read_id='{read_id}'"
                    " has sequence length {seq_len} but only {move_steps} move step(s)"
                    " (non-zero moves={non_zero}, mv_offset={mv_offset}, mv_len={mv_len})".format(
                        root=dataset_root,
                        region=region,
                        read_id=read_id,
                        seq_len=seq_len,
                        move_steps=move_steps,
                        non_zero=non_zero,
                        mv_offset=mv_offset,
                        mv_len=mv_len,
                    )
                )


def _prune_invalid_block_entries(
    *,
    entries_by_region: Dict[str, List[Dict[str, Any]]],
    moves: np.ndarray,
    dataset_root: Path,
) -> int:
    """Drop entries where sequence length exceeds move steps or move steps are zero.

    Returns the number of removed entries.
    """

    removed = 0
    for region in list(entries_by_region.keys()):
        valid_entries: List[Dict[str, Any]] = []
        for entry in entries_by_region[region]:
            mv_offset = entry["mv_offset"]
            mv_len = entry["mv_len"]
            mv_slice = moves[mv_offset : mv_offset + mv_len]
            move_steps = int(np.sum(mv_slice, dtype=np.int64))
            seq_len = len(entry.get("sequence", ""))
            if move_steps <= 0 or seq_len > move_steps:
                removed += 1
                continue
            valid_entries.append(entry)
        if valid_entries:
            entries_by_region[region] = valid_entries
        else:
            entries_by_region.pop(region, None)
    return removed

def _load_block_segments(
    dataset_root: Path,
    *,
    fasta_options: _FastaReferenceOptions,
    allowed_regions: Optional[Set[str]] = None,
) -> List[_SegmentCache]:
    signals_path = dataset_root / "signals.f32.bin"
    moves_path = dataset_root / "mv.u1.bin"
    index_path = dataset_root / "index.tsv"
    fastq_path = dataset_root / "reads.fastq"
    reference_path = dataset_root / "refs.fasta"

    required_paths = (signals_path, moves_path, index_path, fastq_path, reference_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset component missing: {path}")

    available_regions = set(list_block_segments(dataset_root))
    region_filter = allowed_regions or available_regions
    missing = region_filter.difference(available_regions)
    if missing:
        raise ValueError(
            f"Requested region(s) {sorted(missing)} not found in {dataset_root}"
        )

    # Always read full index to preserve global read_id ordering, then filter regions later.
    entries_by_region_all, ordered_entries = _read_block_index(index_path, allowed_regions=None)
    if not entries_by_region_all:
        raise ValueError(f"Index file {index_path} did not yield any entries")

    missing_index_regions = [
        region for region in sorted(region_filter) if region not in entries_by_region_all
    ]
    if missing_index_regions:
        raise KeyError(
            "index.tsv is missing entries for region(s): "
            + ", ".join(missing_index_regions)
        )

    used_read_ids = [entry["read_id"] for entry in ordered_entries]
    read_id_counts = Counter(used_read_ids)
    sequences = _read_fastq_sequences_multimap(
        fastq_path, allowed_read_ids=set(read_id_counts.keys())
    )
    missing_reads = [read_id for read_id in read_id_counts if read_id not in sequences]
    if missing_reads:
        raise KeyError(
            f"FASTQ file {fastq_path} is missing {len(missing_reads)} read(s) referenced in the index"
        )
    # Ensure we have at least as many sequences as required for each read_id
    short_reads = [
        read_id
        for read_id, needed in read_id_counts.items()
        if len(sequences.get(read_id, ())) < needed
    ]
    if short_reads:
        raise ValueError(
            f"FASTQ file {fastq_path} does not contain enough occurrences for read_id(s): {short_reads[:5]}"
        )

    # Attach the correct sequence to each entry following the *global* index order,
    # so duplicate read_ids across regions keep their per-row alignment.
    for entry in ordered_entries:
        read_id = entry["read_id"]
        seq_list = sequences.get(read_id, [])
        if not seq_list:
            raise KeyError(
                f"FASTQ sequence for read '{read_id}' is missing or exhausted for region '{entry['region']}'"
            )
        entry["sequence"] = seq_list.pop(0)

    # Filter entries_by_region after sequences have been attached to preserve alignment.
    entries_by_region = {
        region: entries
        for region, entries in entries_by_region_all.items()
        if region in region_filter
    }

    moves_loader = _lazy_memmap(moves_path, np.uint8)
    moves_mem = moves_loader()
    pruned = _prune_invalid_block_entries(
        entries_by_region=entries_by_region, moves=moves_mem, dataset_root=dataset_root
    )
    if pruned:
        logging.getLogger(__name__).warning(
            "Pruned %d invalid block entr%s under %s (move/base mismatch or zero moves)",
            pruned,
            "y" if pruned == 1 else "ies",
            dataset_root,
        )

    _validate_block_moves(
        entries_by_region=entries_by_region,
        moves_loader=lambda: moves_mem,
        dataset_root=dataset_root,
    )

    reference_records = _parse_fasta(reference_path)
    reference_map = {
        header: _sanitize_reference_sequence(
            sequence,
            path=reference_path,
            policy=fasta_options.ambiguous_base_policy,
        )
        for header, sequence in reference_records
    }

    missing_regions = [region for region in entries_by_region if region not in reference_map]
    if missing_regions:
        raise KeyError(
            "refs.fasta is missing reference sequences for region(s): "
            + ", ".join(sorted(missing_regions))
        )

    signals_loader = _lazy_memmap(signals_path, np.float32)

    segments: List[_SegmentCache] = []
    for region, entries in sorted(entries_by_region.items()):
        read_ids = {entry["read_id"] for entry in entries}
        region_sequences = {
            entry["read_id"]: entry.get("sequence", "") for entry in entries
        }

        reference_string = reference_map[region]
        reference_index = torch.tensor(
            [BASE_VOCAB[base] for base in reference_string], dtype=torch.long
        )

        segments.append(
            _SegmentCache(
                path=dataset_root / region,
                prefix=region,
                signals_path=signals_path,
                moves_path=moves_path,
                sequences=region_sequences,
                entries=entries,
                reference_index=reference_index,
                reference_string=reference_string,
                reference_path=reference_path,
                _signals_loader=signals_loader,
                _moves_loader=moves_loader,
            )
        )

    return segments


def _discover_segments(root: Path) -> List[Path]:
    """Find all segment directories under ``root`` (inclusive)."""

    if _is_segment_directory(root):
        return [root]

    candidates = [child for child in sorted(root.iterdir()) if _is_segment_directory(child)]
    if candidates:
        return candidates

    discovered: Dict[Path, Path] = {}
    for index_path in root.rglob("*.index.tsv"):
        segment_dir = index_path.parent
        if _is_segment_directory(segment_dir):
            discovered[segment_dir.resolve()] = segment_dir

    if not discovered:
        raise FileNotFoundError(
            f"No segment directories found under '{root}'. Expected directories containing"
            " *.index.tsv / *.reads.fastq / *.mv.npy / *.signals.npy files with matching prefixes."
        )

    return sorted(discovered.values(), key=lambda p: str(p))


class ReadDataset(Dataset[Dict[str, Any]]):
    """Dataset yielding all reads for a genomic segment directory."""

    _LOGGER = logging.getLogger("data.read")

    def __init__(
        self,
        segment_path: Path | str,
        *,
        dataset_type: str = "legacy",
        segment_names: Optional[Sequence[str]] = None,
        max_mv_len: Optional[int] = None,
        max_reads_per_segment: Optional[int] = None,
        use_fasta_reference: bool = True,
        fasta_glob_patterns: Optional[Sequence[str]] = None,
        ambiguous_base_policy: str = "error",
        fasta_sequence_policy: str = "first",
        suppress_mv_len_warnings: bool = False,
        use_fastq_base_sequence: bool = True,
        use_read_flag: bool = True,
    ) -> None:
        self.segment_path = Path(segment_path)
        if not self.segment_path.exists():
            raise FileNotFoundError(f"Segment path '{self.segment_path}' does not exist")

        self._dataset_type = str(dataset_type or "legacy").lower()
        allowed_segments = set(segment_names) if segment_names else None

        if not use_fasta_reference:
            raise ValueError(
                "ReadDataset requires FASTA supervision; set use_fasta_reference=True to proceed"
            )

        if ambiguous_base_policy not in AMBIGUOUS_BASE_POLICIES:
            raise ValueError(
                "ambiguous_base_policy must be one of %s" % sorted(AMBIGUOUS_BASE_POLICIES)
            )
        if fasta_sequence_policy not in FASTA_SEQUENCE_POLICIES:
            raise ValueError(
                "fasta_sequence_policy must be one of %s"
                % sorted(FASTA_SEQUENCE_POLICIES)
            )

        patterns = (
            tuple(str(pattern) for pattern in fasta_glob_patterns)
            if fasta_glob_patterns
            else DEFAULT_FASTA_PATTERNS
        )

        self._fasta_options = _FastaReferenceOptions(
            glob_patterns=patterns,
            ambiguous_base_policy=ambiguous_base_policy,
            fasta_sequence_policy=fasta_sequence_policy,
        )

        raw_segments: List[_SegmentCache]
        if self._dataset_type in {"legacy", "npy"}:
            segment_dirs = _discover_segments(self.segment_path)
            if allowed_segments is not None:
                segment_dirs = [
                    path for path in segment_dirs if path.name in allowed_segments
                ]
                if not segment_dirs:
                    raise ValueError(
                        f"Requested segment(s) {sorted(allowed_segments)} not found under '{self.segment_path}'"
                    )
            raw_segments = [
                _load_segment(path, fasta_options=self._fasta_options) for path in segment_dirs
            ]
        elif self._dataset_type in {"blocks", "monolithic"}:
            raw_segments = _load_block_segments(
                self.segment_path,
                fasta_options=self._fasta_options,
                allowed_regions=allowed_segments,
            )
        else:
            raise ValueError(
                "dataset_type must be one of {'legacy', 'npy', 'blocks', 'monolithic'}"
            )

        self._use_fastq_base_sequence = bool(use_fastq_base_sequence)
        if not self._use_fastq_base_sequence:
            self._LOGGER.info(
                "FASTQ base sequences disabled; base_index and base_one_hot will be zero-filled"
            )

        self._use_read_flag = bool(use_read_flag)
        if not self._use_read_flag:
            self._LOGGER.info(
                "Read flag disabled; flag tensor will be zero-filled"
            )

        threshold = int(max_mv_len) if max_mv_len is not None else None
        if threshold is not None and threshold <= 0:
            threshold = None

        read_cap = (
            int(max_reads_per_segment) if max_reads_per_segment is not None else None
        )
        if read_cap is not None and read_cap <= 0:
            raise ValueError("max_reads_per_segment must be a positive integer when provided")

        self._segments: List[_SegmentCache] = []
        self._suppress_mv_len_warnings = bool(suppress_mv_len_warnings)
        skipped_segments = 0

        for segment in raw_segments:
            if threshold is not None:
                exceeding = [entry for entry in segment.entries if entry["mv_len"] > threshold]
                if exceeding:
                    skipped_segments += 1
                    max_len = max(entry["mv_len"] for entry in exceeding)
                    if not self._suppress_mv_len_warnings:
                        self._LOGGER.warning(
                            "Skipping segment '%s' because mv_len=%d exceeds configured max_mv_len=%d",
                            segment.prefix,
                            max_len,
                            threshold,
                        )
                    continue
            if read_cap is not None and len(segment.entries) > read_cap:
                original = len(segment.entries)
                segment.entries = segment.entries[:read_cap]
                self._LOGGER.debug(
                    "Truncated segment '%s' entries from %d to %d due to max_reads_per_segment=%d",
                    segment.prefix,
                    original,
                    len(segment.entries),
                    read_cap,
                )
            self._segments.append(segment)

        if not self._segments:
            if threshold is None:
                raise ValueError(
                    f"ReadDataset found no usable segments under '{self.segment_path}'"
                )
            raise ValueError(
                "ReadDataset filtered out all segments under '%s' with max_mv_len=%d"
                % (self.segment_path, threshold)
            )

        if skipped_segments:
            self._LOGGER.info(
                "Filtered %d segment(s) with mv_len above %d; %d segment(s) remain",
                skipped_segments,
                threshold,
                len(self._segments),
            )

        self.entries: List[Dict[str, Any]] = []
        for segment in self._segments:
            self.entries.extend(entry.copy() for entry in segment.entries)

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        segment = self._segments[index]
        if not segment.entries:
            raise IndexError(f"Segment '{segment.path}' does not contain any entries")

        read_signals: List[torch.Tensor] = []
        read_moves: List[torch.Tensor] = []
        read_base_indices: List[torch.Tensor] = []
        read_base_one_hot: List[torch.Tensor] = []
        read_lengths: List[int] = []
        read_ids: List[str] = []
        read_flags: List[int] = []
        read_strides: List[int] = []

        num_bases = len(BASE_VOCAB)

        signals_memmap = segment.signals_memmap()
        moves_memmap = segment.moves_memmap()

        for entry in segment.entries:
            read_id = entry["read_id"]
            signal_offset = entry["offset"]
            signal_length = entry["length"]
            mv_offset = entry["mv_offset"]
            mv_len = entry["mv_len"]
            stride = entry["stride"]

            signal_slice_np = signals_memmap[signal_offset : signal_offset + signal_length]
            move_slice_np = moves_memmap[mv_offset : mv_offset + mv_len]

            signal_slice = torch.tensor(signal_slice_np, dtype=torch.float32)
            move_slice = torch.tensor(move_slice_np, dtype=torch.float32)

            torch.nan_to_num_(signal_slice, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(move_slice, nan=0.0, posinf=0.0, neginf=0.0)

            move_slice = move_slice.to(torch.long)

            if signal_slice.shape[0] != signal_length:
                raise ValueError("Signal slice length mismatch")
            if move_slice.shape[0] != mv_len:
                raise ValueError("Move slice length mismatch")
            if stride * mv_len != signal_length:
                raise ValueError("Stride, move length and signal length are inconsistent")

            if self._use_fastq_base_sequence:
                # Prefer per-entry sequence (supports duplicate read_ids); fallback to segment map.
                sequence = entry.get("sequence") or segment.sequences.get(read_id)
                if sequence is None:
                    raise KeyError(f"Sequence for read '{read_id}' not found in FASTQ")

                base_indices = _expand_base_sequence(sequence, move_slice.tolist())

                base_one_hot = torch.zeros(mv_len, num_bases, dtype=torch.float32)
                non_blank = base_indices > 0
                if non_blank.any():
                    hot = F.one_hot(
                        (base_indices[non_blank] - 1).to(torch.long), num_classes=num_bases
                    ).to(torch.float32)
                    base_one_hot[non_blank] = hot
            else:
                base_indices = torch.zeros(mv_len, dtype=torch.long)
                base_one_hot = torch.zeros(mv_len, num_bases, dtype=torch.float32)

            read_signals.append(signal_slice.view(mv_len, stride))
            read_moves.append(move_slice.to(torch.float32))
            read_base_indices.append(base_indices)
            read_base_one_hot.append(base_one_hot)
            read_lengths.append(mv_len)
            read_ids.append(read_id)
            if self._use_read_flag:
                read_flags.append(int(entry["flag"]))
            else:
                read_flags.append(0)
            read_strides.append(int(stride))

        if (
            segment.reference_index is None
            or segment.reference_string is None
            or segment.reference_path is None
        ):
            raise ValueError(
                f"Segment '{segment.path}' is missing a FASTA reference; ensure the directory contains a valid FASTA file"
            )

        reference_index = segment.reference_index.clone()
        reference_length = int(reference_index.numel())
        if reference_length <= 0:
            raise ValueError(
                f"Segment '{segment.path}' produced an empty FASTA reference sequence"
            )
        reference_string = segment.reference_string
        reference_path = str(segment.reference_path)

        return {
            "segment_id": segment.prefix,
            "read_id": read_ids,
            "signal": read_signals,
            "move": read_moves,
            "base_index": read_base_indices,
            "base_one_hot": read_base_one_hot,
            "length": read_lengths,
            "stride": read_strides,
            "flag": read_flags,
            "reference_index": reference_index,
            "reference_length": reference_length,
            "reference_string": reference_string,
            "reference_path": reference_path,
        }


def _collate_read_batch_impl(
    batch: Sequence[Dict[str, Any]],
    options: _CollateTransformOptions,
) -> Dict[str, Any]:
    """Pad variable-length multi-read segments into batch tensors."""

    if not batch:
        raise ValueError("collate_read_batch received an empty batch")

    max_reads = max(len(item["signal"]) for item in batch)
    if max_reads == 0:
        raise ValueError("collate_read_batch encountered a segment without reads")

    stride_dim = batch[0]["signal"][0].size(1)
    max_time = 0
    for item in batch:
        for signal in item["signal"]:
            if signal.size(1) != stride_dim:
                raise ValueError("All signals must share the same stride dimension")
            max_time = max(max_time, signal.size(0))

    batch_size = len(batch)
    num_bases = len(BASE_VOCAB)

    signals = torch.zeros(batch_size, max_reads, max_time, stride_dim, dtype=torch.float32)
    moves = torch.zeros(batch_size, max_reads, max_time, dtype=torch.float32)
    base_indices = torch.zeros(batch_size, max_reads, max_time, dtype=torch.long)
    base_one_hots = torch.zeros(batch_size, max_reads, max_time, num_bases, dtype=torch.float32)
    decoder_padding_mask = torch.ones(batch_size, max_reads, max_time, dtype=torch.bool)
    read_padding_mask = torch.ones(batch_size, max_reads, dtype=torch.bool)
    lengths = torch.zeros(batch_size, max_reads, dtype=torch.long)
    strides = torch.zeros(batch_size, max_reads, dtype=torch.long)
    flags = torch.zeros(batch_size, max_reads, dtype=torch.long)

    reference_max_len = 0
    for item in batch:
        ref_tensor = item.get("reference_index")
        if isinstance(ref_tensor, torch.Tensor):
            reference_max_len = max(reference_max_len, int(ref_tensor.numel()))

    if reference_max_len == 0:
        segment_ids = [str(item.get("segment_id", "")) for item in batch]
        raise ValueError(
            "collate_read_batch expected positive-length FASTA references but received empty entries for segments: "
            + ", ".join(segment_ids)
        )

    reference_indices = torch.zeros(batch_size, reference_max_len, dtype=torch.long)
    reference_lengths = torch.zeros(batch_size, dtype=torch.long)

    read_ids: List[List[str]] = []
    segment_ids: List[str] = []
    reference_strings: List[str] = []
    reference_paths: List[str] = []

    for batch_idx, item in enumerate(batch):
        segment_ids.append(item.get("segment_id", ""))
        read_ids.append(item["read_id"])
        for read_idx, (signal, move, base_index, base_one_hot) in enumerate(
            zip(item["signal"], item["move"], item["base_index"], item["base_one_hot"])
        ):
            length = signal.size(0)
            signals[batch_idx, read_idx, :length] = signal
            moves[batch_idx, read_idx, :length] = move
            base_indices[batch_idx, read_idx, :length] = base_index
            base_one_hots[batch_idx, read_idx, :length] = base_one_hot
            decoder_padding_mask[batch_idx, read_idx, :length] = False
            read_padding_mask[batch_idx, read_idx] = False
            lengths[batch_idx, read_idx] = length
            strides[batch_idx, read_idx] = int(item["stride"][read_idx])
            flags[batch_idx, read_idx] = int(item["flag"][read_idx])

        ref_tensor = item.get("reference_index")
        ref_length = int(item.get("reference_length", 0) or 0)
        if not isinstance(ref_tensor, torch.Tensor):
            raise ValueError("Batch item is missing 'reference_index' tensor")

        if ref_length <= 0:
            raise ValueError(
                f"Batch item for segment '{segment_ids[-1]}' reports non-positive reference_length={ref_length}"
            )

        if reference_max_len > 0:
            trimmed = ref_tensor[:ref_length]
            if trimmed.numel() != ref_length:
                raise ValueError("reference_index length does not match reference_length")
            reference_indices[batch_idx, :ref_length] = trimmed
        reference_lengths[batch_idx] = ref_length
        reference_strings.append(str(item.get("reference_string", "")))
        reference_paths.append(str(item.get("reference_path", "")))

    _apply_collate_transforms(signals, lengths, read_padding_mask, options)

    batch_dict: Dict[str, Any] = {
        "segment_id": segment_ids,
        "read_id": read_ids,
        "signal": signals,
        "move": moves,
        "base_index": base_indices,
        "base_one_hot": base_one_hots,
        "decoder_padding_mask": decoder_padding_mask,
        "read_padding_mask": read_padding_mask,
        "length": lengths,
        "stride": strides,
        "flag": flags,
        "reference_index": reference_indices,
        "reference_lengths": reference_lengths,
        "reference_string": reference_strings,
        "reference_path": reference_paths,
    }

    batch_dict["hard_mask"] = (~decoder_padding_mask).to(torch.float32)
    return batch_dict


def collate_read_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate function without additional transforms."""

    return _collate_read_batch_impl(batch, _DEFAULT_COLLATE_OPTIONS)


def build_collate_read_batch(
    config: Mapping[str, Any] | _CollateTransformOptions | None = None,
):
    """Create a collate function with optional normalization/augmentation."""

    options = _parse_collate_transform_config(config)
    if options == _DEFAULT_COLLATE_OPTIONS:
        return collate_read_batch
    return functools.partial(_collate_read_batch_impl, options=options)


@dataclass
class DataModule:
    """Lightweight substitute for Lightning-style data modules."""

    train_dataset: Dataset[Any]
    val_dataset: Optional[Dataset[Any]] = None
    test_dataset: Optional[Dataset[Any]] = None
    batch_size: int = 1
    num_workers: int = 0
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    shuffle: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    train_collate_config: Optional[Mapping[str, Any]] = None
    val_collate_config: Optional[Mapping[str, Any]] = None
    test_collate_config: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        self._train_collate_fn = build_collate_read_batch(self.train_collate_config)
        self._val_collate_fn = build_collate_read_batch(self.val_collate_config)
        self._test_collate_fn = build_collate_read_batch(self.test_collate_config)

    def _make_loader(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        shuffle: bool,
        collate_fn,
    ) -> DataLoader[Any]:
        loader_kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": self.num_workers,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "pin_memory": self.pin_memory,
        }

        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._make_loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader[Any]]:
        if self.val_dataset is None:
            return None
        return self._make_loader(
            self.val_dataset,
            batch_size=self.val_batch_size or self.batch_size,
            shuffle=False,
            collate_fn=self._val_collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader[Any]]:
        if self.test_dataset is None:
            return None
        return self._make_loader(
            self.test_dataset,
            batch_size=self.test_batch_size or self.batch_size,
            shuffle=False,
            collate_fn=self._test_collate_fn,
        )


__all__ = [
    "DataModule",
    "ReadDataset",
    "collate_read_batch",
    "build_collate_read_batch",
    "list_block_segments",
]
