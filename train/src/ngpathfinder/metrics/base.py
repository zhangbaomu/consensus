"""Metric stubs."""
from __future__ import annotations

from typing import Dict

import torch


class Metric:
    """Base metric interface."""

    def update(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class RunningAverage(Metric):
    """Simple running loss metric."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:  # noqa: D401
        value = float(outputs.get("loss", 0.0))
        self.total += value
        self.count += 1

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"running_average": 0.0}
        return {"running_average": self.total / self.count}

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


__all__ = ["Metric", "RunningAverage"]
