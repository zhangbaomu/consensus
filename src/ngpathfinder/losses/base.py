"""Loss registry and placeholders."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from ..registry import get_registry

LOSS_REGISTRY = get_registry("loss")


class LossWrapper(nn.Module):
    """Wrap torch loss modules for registry integration."""

    def __init__(self, criterion: nn.Module) -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "logits" not in outputs or "targets" not in batch:
            raise KeyError("Outputs must contain 'logits' and batch must contain 'targets'")
        return self.criterion(outputs["logits"], batch["targets"])


@LOSS_REGISTRY.register("mse")
def build_mse_loss(**params: Any) -> nn.Module:
    return LossWrapper(nn.MSELoss(**params))


__all__ = ["LOSS_REGISTRY", "LossWrapper"]
