"""Utilities for validating signal datasets.

This module implements reusable helpers that can be imported both from
command line tools and from data processing pipelines.  The current
validation checks follow the data description provided in the project
documentation:

* ``mv_len * stride`` must be equal to ``length`` for every read.  In the
  provided datasets the stride is expected to be ``6`` which results in the
  previously mentioned ``6 * mv_len = length`` invariant.
* The number of ``1`` values inside each read slice of the move table must
  be equal to the length of the basecalled sequence found in the FASTQ file.
* ``offset``/``length`` pairs for the signal array must be within bounds and
  the concatenation of all read slices must exactly consume the move-table and
  signal arrays (no unused prefix/suffix data and no gaps/overlaps between
  adjacent reads).

When new checks are required they can be added here so that both the
``check_dataset`` command line interface and any future pipelines can reuse
the same logic without code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np


@dataclass
class ValidationIssue:
    """Represents a failed validation check for a specific read."""

    segment: Path
    read_id: str
    code: str
    message: str

    def format(self) -> str:
        """Return a human readable representation of the issue."""

        relative_segment = self.segment.name or str(self.segment)
        return (
            f"[{self.code}] segment={relative_segment} read_id={self.read_id}: "
            f"{self.message}"
        )


def load_fastq_sequence_lengths(fastq_path: Path) -> Dict[str, int]:
    """Return a mapping from read id to sequence length for a FASTQ file."""

    lengths: Dict[str, int] = {}
    with fastq_path.open("r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()

            if not (sequence and plus and quality):
                raise ValueError(
                    f"FASTQ file {fastq_path} is truncated or malformed."
                )

            if not header.startswith("@"):
                raise ValueError(
                    f"Invalid FASTQ header in {fastq_path}: {header!r}"
                )

            read_id = header[1:].strip()
            lengths[read_id] = len(sequence.strip())

    return lengths


def _read_index_rows(index_path: Path) -> Iterator[Dict[str, str]]:
    """Yield dictionaries for each row in an index TSV file."""

    with index_path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"Index file {index_path} is empty")

        headers = header_line.strip().split()

        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != len(headers):
                raise ValueError(
                    "Index row has unexpected number of columns: "
                    f"expected {len(headers)}, got {len(parts)}"
                )
            yield dict(zip(headers, parts))


def validate_segment(segment_dir: Path) -> List[ValidationIssue]:
    """Run validation checks for a single segment directory.

    Parameters
    ----------
    segment_dir:
        Directory containing ``*.index.tsv``, ``*.reads.fastq`` and
        ``*.mv.npy`` files.
    """

    if not segment_dir.is_dir():
        raise FileNotFoundError(f"Segment directory {segment_dir} does not exist")

    index_paths = list(segment_dir.glob("*.index.tsv"))
    if not index_paths:
        raise FileNotFoundError(f"No index TSV found in {segment_dir}")
    if len(index_paths) > 1:
        raise ValueError(
            f"Multiple index TSV files found in {segment_dir}: {index_paths}"
        )
    index_path = index_paths[0]

    fastq_paths = list(segment_dir.glob("*.reads.fastq"))
    if not fastq_paths:
        raise FileNotFoundError(f"No FASTQ file found in {segment_dir}")
    if len(fastq_paths) > 1:
        raise ValueError(
            f"Multiple FASTQ files found in {segment_dir}: {fastq_paths}"
        )
    fastq_path = fastq_paths[0]

    mv_paths = list(segment_dir.glob("*.mv.npy"))
    if not mv_paths:
        raise FileNotFoundError(f"No move table (.mv.npy) found in {segment_dir}")
    if len(mv_paths) > 1:
        raise ValueError(
            f"Multiple move tables found in {segment_dir}: {mv_paths}"
        )
    mv_path = mv_paths[0]

    mv_array = np.load(mv_path)

    signal_paths = list(segment_dir.glob("*.signals.npy"))
    if not signal_paths:
        raise FileNotFoundError(
            f"No signal array (.signals.npy) found in {segment_dir}"
        )
    if len(signal_paths) > 1:
        raise ValueError(
            f"Multiple signal arrays found in {segment_dir}: {signal_paths}"
        )
    signal_path = signal_paths[0]
    signal_array = np.load(signal_path)

    fastq_lengths = load_fastq_sequence_lengths(fastq_path)

    issues: List[ValidationIssue] = []

    mv_ranges: List[Tuple[int, int, str]] = []
    mv_has_oob = False

    signal_ranges: List[Tuple[int, int, str]] = []
    signal_has_oob = False

    for row in _read_index_rows(index_path):
        read_id = row["read_id"]
        try:
            mv_offset = int(row["mv_offset"])
            mv_len = int(row["mv_len"])
            stride = int(row.get("stride", 6))
            length = int(row["length"])
            signal_offset = int(row["offset"])
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Missing column {exc} in index file {index_path}")

        # Check mv_len * stride == length (defaulting to 6 if stride is missing).
        expected_signal_len = mv_len * stride
        if expected_signal_len != length:
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id=read_id,
                    code="stride_mismatch",
                    message=(
                        f"mv_len({mv_len}) * stride({stride}) = {expected_signal_len} "
                        f"!= length({length})"
                    ),
                )
            )

        mv_end = mv_offset + mv_len

        mv_range_valid = True
        if mv_offset < 0:
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id=read_id,
                    code="mv_offset_negative",
                    message=f"mv_offset {mv_offset} is negative",
                )
            )
            mv_range_valid = False
        if mv_len < 0:
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id=read_id,
                    code="mv_len_negative",
                    message=f"mv_len {mv_len} is negative",
                )
            )
            mv_range_valid = False

        if mv_range_valid:
            mv_ranges.append((mv_offset, mv_end, read_id))

            if mv_end > len(mv_array):
                issues.append(
                    ValidationIssue(
                        segment=segment_dir,
                        read_id=read_id,
                        code="mv_slice_oob",
                        message=(
                            f"mv slice [{mv_offset}:{mv_end}] exceeds mv array size {len(mv_array)}"
                        ),
                    )
                )
                mv_has_oob = True
                mv_range_valid = False

        if mv_range_valid:
            mv_slice = mv_array[mv_offset:mv_end]
            if mv_slice.shape[0] != mv_len:
                issues.append(
                    ValidationIssue(
                        segment=segment_dir,
                        read_id=read_id,
                        code="mv_slice_len",
                        message=(
                            f"mv slice length {mv_slice.shape[0]} != mv_len ({mv_len})"
                        ),
                    )
                )

            ones_count = int(np.count_nonzero(mv_slice))
            seq_len = fastq_lengths.get(read_id)

            if seq_len is None:
                issues.append(
                    ValidationIssue(
                        segment=segment_dir,
                        read_id=read_id,
                        code="missing_fastq",
                        message="Read id not found in FASTQ file",
                    )
                )
            elif seq_len != ones_count:
                issues.append(
                    ValidationIssue(
                        segment=segment_dir,
                        read_id=read_id,
                        code="sequence_length_mismatch",
                        message=(
                            f"FASTQ sequence length {seq_len} != count_ones(mv_slice) {ones_count}"
                        ),
                    )
                )

        signal_end = signal_offset + length
        signal_range_valid = True
        if signal_offset < 0:
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id=read_id,
                    code="signal_offset_negative",
                    message=f"offset {signal_offset} is negative",
                )
            )
            signal_range_valid = False
        if length < 0:
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id=read_id,
                    code="signal_length_negative",
                    message=f"length {length} is negative",
                )
            )
            signal_range_valid = False

        if signal_range_valid:
            signal_ranges.append((signal_offset, signal_end, read_id))

            if signal_end > len(signal_array):
                issues.append(
                    ValidationIssue(
                        segment=segment_dir,
                        read_id=read_id,
                        code="signal_slice_oob",
                        message=(
                            "signal slice "
                            f"[{signal_offset}:{signal_end}] exceeds signal array size {len(signal_array)}"
                        ),
                    )
                )
                signal_has_oob = True
                signal_range_valid = False

        if signal_range_valid:
            signal_slice = signal_array[signal_offset:signal_end]
            if signal_slice.shape[0] != length:
                issues.append(
                    ValidationIssue(
                        segment=segment_dir,
                        read_id=read_id,
                        code="signal_slice_len",
                        message=(
                            f"signal slice length {signal_slice.shape[0]} != length ({length})"
                        ),
                    )
                )

    if mv_ranges and not mv_has_oob:
        issues.extend(
            _check_contiguous_ranges(
                ranges=mv_ranges,
                array_length=len(mv_array),
                segment_dir=segment_dir,
                prefix_code="mv_unused_prefix",
                tail_code="mv_unused_tail",
                gap_code="mv_gap",
                overlap_code="mv_overlap",
                label="mv",
            )
        )

    if signal_ranges and not signal_has_oob:
        issues.extend(
            _check_contiguous_ranges(
                ranges=signal_ranges,
                array_length=len(signal_array),
                segment_dir=segment_dir,
                prefix_code="signal_unused_prefix",
                tail_code="signal_unused_tail",
                gap_code="signal_gap",
                overlap_code="signal_overlap",
                label="signal",
            )
        )

    return issues


def _check_contiguous_ranges(
    *,
    ranges: Sequence[Tuple[int, int, str]],
    array_length: int,
    segment_dir: Path,
    prefix_code: str,
    tail_code: str,
    gap_code: str,
    overlap_code: str,
    label: str,
) -> List[ValidationIssue]:
    """Ensure slices exactly cover an array without gaps or overlaps."""

    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda item: (item[0], item[1]))

    issues: List[ValidationIssue] = []

    first_offset, first_end, first_read = sorted_ranges[0]
    if first_offset > 0:
        issues.append(
            ValidationIssue(
                segment=segment_dir,
                read_id="<segment>",
                code=prefix_code,
                message=(
                    f"{label} array has unused prefix of length {first_offset} "
                    f"before read {first_read}"
                ),
            )
        )

    prev_end = first_end
    prev_read = first_read

    for offset, end, read_id in sorted_ranges[1:]:
        if offset > prev_end:
            gap = offset - prev_end
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id="<segment>",
                    code=gap_code,
                    message=(
                        f"{label} gap of {gap} between reads {prev_read} (end={prev_end}) "
                        f"and {read_id} (start={offset})"
                    ),
                )
            )
            prev_end = end
            prev_read = read_id
            continue

        if offset < prev_end:
            overlap = prev_end - offset
            issues.append(
                ValidationIssue(
                    segment=segment_dir,
                    read_id="<segment>",
                    code=overlap_code,
                    message=(
                        f"{label} overlap of {overlap} between reads {prev_read} "
                        f"(end={prev_end}) and {read_id} (start={offset})"
                    ),
                )
            )
            if end > prev_end:
                prev_end = end
                prev_read = read_id
            continue

        # offset == prev_end (perfect adjacency)
        prev_end = end
        prev_read = read_id

    if prev_end < array_length:
        issues.append(
            ValidationIssue(
                segment=segment_dir,
                read_id="<segment>",
                code=tail_code,
                message=(
                    f"{label} array has unused tail of length {array_length - prev_end} "
                    f"after read {prev_read}"
                ),
            )
        )

    return issues


def iter_segment_directories(dataset_path: Path) -> Iterator[Path]:
    """Yield segment directories contained in ``dataset_path``.

    ``dataset_path`` can either point directly at a segment directory or at a
    parent directory containing one or more segments.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

    if any(dataset_path.glob("*.index.tsv")):
        yield dataset_path
        return

    for child in sorted(dataset_path.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.index.tsv")):
            yield child

