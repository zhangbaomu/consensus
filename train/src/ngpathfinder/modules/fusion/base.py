"""Fusion module definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn

from ...registry import get_registry

FUSION_REGISTRY = get_registry("fusion")


class FusionBase(nn.Module, ABC):
    """Combine multiple read encodings using learnable queries."""

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return fused representations with fixed-length queries."""


class IdentityFusion(FusionBase):
    """By default, do nothing."""

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch


@FUSION_REGISTRY.register("identity")
def build_identity_fusion(**_: Any) -> FusionBase:
    return IdentityFusion()


__all__ = ["FusionBase", "FUSION_REGISTRY"]
