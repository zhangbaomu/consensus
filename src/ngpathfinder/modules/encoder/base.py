"""Base classes and default encoder implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn

from ...registry import get_registry

ENCODER_REGISTRY = get_registry("encoder")


class EncoderBase(nn.Module, ABC):
    """Abstract encoder that processes per-read signal chunks."""

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode raw input batch into embeddings."""


class IdentityEncoder(EncoderBase):
    """Minimal encoder that forwards inputs untouched."""

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch


@ENCODER_REGISTRY.register("identity")
def build_identity_encoder(**_: Any) -> EncoderBase:
    return IdentityEncoder()


__all__ = ["EncoderBase", "IdentityEncoder", "ENCODER_REGISTRY"]
