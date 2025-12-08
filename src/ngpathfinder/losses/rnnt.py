"""RNN-T loss wrapper."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from .base import LOSS_REGISTRY
from .ctc_common import _gather_reference
from .ctc_constants import NUM_BASES

try:  # pragma: no cover - optional dependency
    from torchaudio.functional import rnnt_loss as _rnnt_loss
except Exception:  # pragma: no cover
    _rnnt_loss = None


BLANK_INDEX = 0
VOCAB_SIZE = NUM_BASES + 1


def _pad_targets(targets: Tensor, lengths: Tensor) -> Tensor:
    if targets.dim() != 2:
        raise ValueError("targets must have shape (B, U)")
    if lengths.dim() != 1 or lengths.numel() != targets.size(0):
        raise ValueError("lengths must be 1D and align with targets batch")

    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    padded = targets.new_full((targets.size(0), max_len), BLANK_INDEX)
    for idx in range(targets.size(0)):
        length = int(lengths[idx].item())
        if length <= 0:
            raise ValueError("target lengths must be positive for RNNT loss")
        if length > targets.size(1):
            raise ValueError("target length exceeds provided target tensor width")
        padded[idx, :length] = targets[idx, :length]
    return padded


class RNNTLoss(nn.Module):
    """Compute RNN-T loss from decoder logits."""

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        if _rnnt_loss is None:
            raise ImportError(
                "torchaudio.functional.rnnt_loss is unavailable; install torchaudio with RNNT support"
            )
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")
        self.reduction = reduction

    def _resolve_logit_lengths(
        self, logits: Tensor, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_size, time_steps, _, _ = logits.shape
        logit_lengths = outputs.get("rnnt_logit_lengths")
        if logit_lengths is not None:
            if logit_lengths.dim() != 1 or logit_lengths.numel() != batch_size:
                raise ValueError("rnnt_logit_lengths must be 1D and match the batch dimension")
            return logit_lengths.to(dtype=torch.int32, device=logits.device), torch.empty(0)

        pad_mask = outputs.get("decoder_padding_mask")
        if pad_mask is None:
            pad_mask = batch.get("decoder_padding_mask")
        if pad_mask is not None:
            if pad_mask.dim() == 3:
                pad_mask = pad_mask.all(dim=1)
            if pad_mask.dim() != 2:
                raise ValueError("decoder_padding_mask must have shape (B, T) or (B, R, T)")
            if pad_mask.size(0) != batch_size:
                raise ValueError("decoder_padding_mask batch dimension must match logits")
            if pad_mask.size(1) != time_steps:
                if pad_mask.size(1) > time_steps:
                    pad_mask = pad_mask[:, :time_steps]
                else:
                    pad = torch.ones(
                        pad_mask.size(0), time_steps - pad_mask.size(1), dtype=pad_mask.dtype, device=pad_mask.device
                    )
                    pad_mask = torch.cat((pad_mask, pad), dim=1)
            lengths = (~pad_mask.to(device=logits.device, dtype=torch.bool)).sum(dim=1)
            return lengths.to(dtype=torch.int32), pad_mask

        fallback = torch.full((batch_size,), time_steps, dtype=torch.int32, device=logits.device)
        return fallback, torch.empty(0)

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        if "rnnt_logits" not in outputs:
            raise KeyError("Decoder outputs must contain 'rnnt_logits'")
        logits = outputs["rnnt_logits"]
        if logits.dim() != 4:
            raise ValueError("rnnt_logits must have shape (B, T, U, V)")
        batch_size, _, _, vocab_size = logits.shape
        if vocab_size != VOCAB_SIZE:
            raise ValueError(f"rnnt_logits last dimension must be {VOCAB_SIZE}, received {vocab_size}")

        logit_lengths, _ = self._resolve_logit_lengths(logits, outputs, batch)

        targets, target_lengths = _gather_reference(batch)
        if len(targets) != batch_size:
            raise ValueError("Gathered targets do not match batch size")

        max_len = max((t.numel() for t in targets), default=0)
        target_tensor = torch.full(
            (batch_size, max_len), BLANK_INDEX, device=logits.device, dtype=torch.int32
        )
        for idx, (tgt, tgt_len) in enumerate(zip(targets, target_lengths.tolist())):
            if tgt_len <= 0:
                raise ValueError("RNNT loss requires positive target lengths")
            if tgt_len > tgt.numel():
                raise ValueError("target_lengths entry exceeds target tensor length")
            target_tensor[idx, :tgt_len] = tgt.to(device=logits.device, dtype=torch.int32)

        target_lengths = target_lengths.to(dtype=torch.int32, device=logits.device)
        padded_targets = _pad_targets(target_tensor, target_lengths)

        if logits.dtype not in (torch.float16, torch.float32):
            logits = logits.float()

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss = _rnnt_loss(
            logits,
            padded_targets,
            logit_lengths.to(dtype=torch.int32, device=logits.device),
            target_lengths.to(dtype=torch.int32, device=logits.device),
            blank=BLANK_INDEX,
            reduction=self.reduction,
        )

        return loss


@LOSS_REGISTRY.register("rnnt")
def build_rnnt_loss(**kwargs: object) -> nn.Module:
    return RNNTLoss(**kwargs)


__all__ = ["RNNTLoss", "build_rnnt_loss"]