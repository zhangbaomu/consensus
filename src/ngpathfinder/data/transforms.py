"""Signal preprocessing and augmentation stubs."""
from __future__ import annotations

from typing import Dict

import torch


def slice_signal(signal: torch.Tensor, window: int = 6) -> torch.Tensor:
    """Placeholder slicing routine."""
    if signal.numel() == 0:
        return signal
    length = (signal.shape[-1] // window) * window
    return signal[..., :length].view(*signal.shape[:-1], -1, window)


def attach_metadata(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stub for concatenating move/basecall metadata to slices."""
    return batch


__all__ = ["slice_signal", "attach_metadata"]
