"""Decoder definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn

from ...registry import get_registry

DECODER_REGISTRY = get_registry("decoder")


class DecoderBase(nn.Module, ABC):
    """Temporal model producing nucleotide logits."""

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run decoder over aggregated features."""


class IdentityDecoder(DecoderBase):
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch


@DECODER_REGISTRY.register("identity")
def build_identity_decoder(**_: Any) -> DecoderBase:
    return IdentityDecoder()


__all__ = ["DecoderBase", "DECODER_REGISTRY"]
