"""Aggregator definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn

from ...registry import get_registry

AGGREGATOR_REGISTRY = get_registry("aggregator")


class AggregatorBase(nn.Module, ABC):
    """Aggregate fused representations into decoder inputs."""

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aggregate fused sequence representations."""


class IdentityAggregator(AggregatorBase):
    """Pass tensors through when a single read is present."""

    _THREE_D_KEYS = {
        "decoder_padding_mask",
        "hard_mask",
        "soft_hint",
        "move",
        "base_index",
    }

    _TWO_D_KEYS = {"length", "stride", "flag"}

    _FOUR_D_KEYS = {"base_one_hot"}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedding = batch.get("embedding")
        if embedding is None:
            raise KeyError("IdentityAggregator requires 'embedding' in batch")
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("'embedding' must be a torch.Tensor")

        if embedding.dim() != 4:
            return batch

        _, num_reads, _, _ = embedding.shape
        if num_reads > 1:
            raise ValueError(
                "IdentityAggregator only supports batches with a single read per segment; "
                "configure max_reads_per_segment=1 or switch to a learned aggregator"
            )

        squeezed: Dict[str, torch.Tensor] = dict(batch)
        squeezed["embedding"] = embedding[:, 0]

        for key in self._THREE_D_KEYS:
            value = squeezed.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                squeezed[key] = value[:, 0]

        for key in self._TWO_D_KEYS:
            value = squeezed.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 2:
                squeezed[key] = value[:, 0]

        for key in self._FOUR_D_KEYS:
            value = squeezed.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 4:
                squeezed[key] = value[:, 0]

        read_mask = squeezed.get("read_padding_mask")
        if isinstance(read_mask, torch.Tensor):
            if read_mask.dim() == 2:
                squeezed["read_padding_mask"] = read_mask[:, 0]
            elif read_mask.dim() not in {0, 1}:
                raise ValueError(
                    "IdentityAggregator expects read_padding_mask to be 1D when only a single read is present"
                )

        return squeezed


@AGGREGATOR_REGISTRY.register("identity")
def build_identity_aggregator(**_: Any) -> AggregatorBase:
    return IdentityAggregator()


__all__ = ["AggregatorBase", "AGGREGATOR_REGISTRY"]