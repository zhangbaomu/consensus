"""Temporal order regularization for fusion queries."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import LOSS_REGISTRY


class TemporalOrderLoss(nn.Module):
    """Encourage monotonic progression of fusion query positions."""

    def __init__(self, weight: float = 0.2, margin: float = 0.0) -> None:
        super().__init__()
        self.weight = float(weight)
        self.margin = float(margin)

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        pos_rq: Optional[Tensor] = None
        weights: Optional[Tensor] = None
        pad: Optional[Tensor] = None

        if isinstance(outputs, dict):
            pos_rq = outputs.get("fusion_soft_position")
            weights = outputs.get("fusion_read_weights")
            pad = outputs.get("decoder_padding_mask")

        if pos_rq is None and isinstance(batch, dict):
            pos_rq = batch.get("fusion_soft_position")
        if weights is None and isinstance(batch, dict):
            weights = batch.get("fusion_read_weights")
        if pad is None and isinstance(batch, dict):
            pad = batch.get("decoder_padding_mask")

        reference: Optional[Tensor] = None
        if isinstance(outputs, dict):
            maybe_logits = outputs.get("ctc_logits")
            if isinstance(maybe_logits, Tensor):
                reference = maybe_logits
            else:
                for value in outputs.values():
                    if isinstance(value, Tensor):
                        reference = value
                        break

        if reference is None and isinstance(pos_rq, Tensor):
            reference = pos_rq
        if reference is None and isinstance(weights, Tensor):
            reference = weights

        if pos_rq is None or weights is None:
            if reference is not None:
                return reference.new_zeros(())
            return torch.zeros(())

        if pad is None:
            pad = torch.zeros(pos_rq.size(0), pos_rq.size(-1), dtype=torch.bool, device=pos_rq.device)

        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pos = (pos_rq * weights.unsqueeze(-1)).sum(dim=1) / weight_sum
        valid = (~pad).to(pos.dtype)

        if pos.size(1) <= 1:
            return pos.sum() * 0.0

        diff = pos[:, 1:] - pos[:, :-1]
        pair_mask = valid[:, 1:] * valid[:, :-1]
        pair_mask = pair_mask.to(pos.dtype)
        viol = F.relu(self.margin - diff) * pair_mask

        denom = pair_mask.sum().clamp_min(1.0)
        loss = viol.sum() / denom
        return loss * self.weight


__all__ = ["TemporalOrderLoss"]


@LOSS_REGISTRY.register("temporal_order")
def build_temporal_order_loss(**kwargs) -> nn.Module:
    return TemporalOrderLoss(
        weight=float(kwargs.get("weight", 0.2)),
        margin=float(kwargs.get("margin", 0.0)),
    )