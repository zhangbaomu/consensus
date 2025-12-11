"""Auxiliary losses for decoder duration heads."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import LOSS_REGISTRY


def _gather_valid_steps(
    tensor: Tensor, gather_indices: Tensor, lengths: Tensor
) -> Tensor:
    if gather_indices.dim() != 2:
        raise ValueError("ctc_logit_gather_indices must have shape (B, T)")
    if gather_indices.size(0) != tensor.size(0):
        raise ValueError("gather_indices batch dimension must match tensor batch size")

    batch, max_valid = gather_indices.shape
    compact = tensor.new_zeros(batch, max_valid)
    for batch_idx in range(batch):
        valid_len = int(lengths[batch_idx].item()) if batch_idx < lengths.numel() else 0
        if valid_len <= 0:
            continue
        step_indices = gather_indices[batch_idx, :valid_len]
        compact[batch_idx, :valid_len] = tensor[batch_idx].index_select(0, step_indices)
    return compact


class DurationBCELoss(nn.Module):
    """Binary cross-entropy loss for duration logits with masking support."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("DurationBCELoss weight must be non-negative")
        self.weight = float(weight)

    def forward(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tensor:  # type: ignore[override]
        if "duration_logits" not in outputs:
            raise KeyError("Decoder outputs must contain 'duration_logits'")
        logits = outputs["duration_logits"]
        if logits.dim() != 2:
            raise ValueError("duration_logits must have shape (B, T)")

        lengths = outputs.get("ctc_logit_lengths")
        gather_indices = outputs.get("ctc_logit_gather_indices")
        if not isinstance(lengths, Tensor) or not isinstance(gather_indices, Tensor):
            raise KeyError(
                "Decoder outputs must include 'ctc_logit_lengths' and 'ctc_logit_gather_indices'"
            )

        target = outputs.get("duration_target")
        if target is None:
            target = batch.get("duration_target")
        if target is None:
            raise KeyError(
                "Batch or decoder outputs must provide 'duration_target' when using DurationBCELoss"
            )
        if not isinstance(target, Tensor):
            raise TypeError("duration_target must be a torch.Tensor")
        if target.dim() != 2:
            raise ValueError("duration_target must have shape (B, T)")
        if target.size(0) != logits.size(0):
            raise ValueError("duration_target batch dimension must match duration_logits")

        gathered_target = _gather_valid_steps(target, gather_indices, lengths)
        if gathered_target.shape != logits.shape:
            raise ValueError("Gathered duration target shape must match duration_logits")

        loss = F.binary_cross_entropy_with_logits(
            logits,
            gathered_target,
            reduction="none",
        )

        batch_size, max_valid = logits.shape
        device = logits.device
        mask = torch.arange(max_valid, device=device).view(1, -1) < lengths.view(-1, 1)
        masked_loss = loss * mask.to(loss.dtype)
        normalizer = mask.sum().clamp_min(1).to(loss.dtype)
        mean_loss = masked_loss.sum() / normalizer
        return mean_loss * self.weight


__all__ = ["DurationBCELoss"]


@LOSS_REGISTRY.register("duration_bce")
def build_duration_bce_loss(**kwargs: float) -> nn.Module:
    weight = float(kwargs.get("weight", 1.0))
    return DurationBCELoss(weight=weight)