"""Shared utilities for CTC-CRF style losses."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def minimal_ctc_input_length(target: Tensor, valid_length: int | None = None) -> int:
    """Return the minimal decoder steps required for a CTC target."""

    if valid_length is None:
        valid_length = int(target.numel())
    else:
        valid_length = int(valid_length)

    if valid_length <= 1:
        return max(valid_length, 0)

    trimmed = target[:valid_length]
    repeats = (trimmed[1:] == trimmed[:-1]).sum().item()
    return valid_length + int(repeats)


def _gather_reference(batch: Dict[str, Tensor]) -> Tuple[List[Tensor], Tensor]:
    reference_index = batch.get("reference_index")
    reference_lengths = batch.get("reference_lengths")

    if not isinstance(reference_index, Tensor) or not isinstance(reference_lengths, Tensor):
        raise KeyError(
            "CTC-CRF loss expects 'reference_index' and 'reference_lengths' tensors sourced from FASTA references"
        )

    reference_index = reference_index.long()
    reference_lengths = reference_lengths.to(dtype=torch.long)

    if reference_index.dim() == 1:
        reference_index = reference_index.unsqueeze(0)
    if reference_index.dim() != 2:
        raise ValueError("'reference_index' must have shape (batch, time)")

    if reference_lengths.dim() == 0:
        reference_lengths = reference_lengths.unsqueeze(0)
    if reference_lengths.dim() != 1:
        raise ValueError("'reference_lengths' must have shape (batch,)")
    if reference_lengths.numel() != reference_index.size(0):
        raise ValueError("reference_lengths must have the same batch dimension as reference_index")

    targets: List[Tensor] = []
    for batch_idx in range(reference_index.size(0)):
        length = int(reference_lengths[batch_idx].item())
        if length <= 0:
            segment_ids = batch.get("segment_id")
            segment_msg = ""
            if isinstance(segment_ids, list) and batch_idx < len(segment_ids):
                segment_msg = f" for segment '{segment_ids[batch_idx]}'"
            raise ValueError(
                f"reference_lengths must be positive{segment_msg}; received {length}"
            )
        if length > reference_index.size(1):
            raise ValueError("reference_length exceeds available reference_index entries")
        targets.append(reference_index[batch_idx, :length])

    return targets, reference_lengths


__all__ = ["minimal_ctc_input_length", "_gather_reference"]