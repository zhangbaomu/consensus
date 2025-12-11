"""Fast-path loss that wraps PyTorch's native CTC criterion."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .base import LOSS_REGISTRY
from .ctc_common import _gather_reference
from .ctc_constants import NUM_BASES

BLANK_INDEX = 0
EXPECTED_VOCAB = NUM_BASES + 1


def minimal_ctc_input_length(target: Tensor, valid_length: int | None = None) -> int:
    """Return the minimal number of decoder steps required for a CTC target.

    Args:
        target: Tensor containing the encoded reference sequence.
        valid_length: Optional length specifying how many leading entries of
            ``target`` constitute the reference. When omitted, the entire tensor
            is considered valid.

    Returns:
        Minimal number of decoder steps necessary for the reference when
        accounting for repeated bases that require interleaved blanks.
    """

    if valid_length is None:
        valid_length = int(target.numel())
    else:
        valid_length = int(valid_length)

    if valid_length <= 1:
        return max(valid_length, 0)

    trimmed = target[:valid_length]
    repeats = (trimmed[1:] == trimmed[:-1]).sum().item()
    return valid_length + int(repeats)


def _max_run_length(sequence: Tensor) -> int:
    """Return the longest run-length of identical labels within ``sequence``."""

    length = int(sequence.numel())
    if length == 0:
        return 0

    # ``sequence`` is often a CUDA tensor; convert once to CPU scalars to avoid
    # synchronising on every element via ``.item()`` during iteration.
    values = sequence.detach().to("cpu").tolist()

    max_run = 1
    current = 1
    prev = values[0]
    for value in values[1:]:
        if value == prev:
            current += 1
        else:
            if current > max_run:
                max_run = current
            current = 1
            prev = value
    if current > max_run:
        max_run = current
    return max_run


def _compute_hpoly_weights(
    targets: List[Tensor],
    *,
    min_run: int,
    cap_run: int,
    max_boost: float,
) -> Tensor:
    """Compute homopolymer-aware weights for each sample in ``targets``."""

    if cap_run < min_run:
        raise ValueError("cap_run must be greater than or equal to min_run")
    if max_boost < 1.0:
        raise ValueError("max_boost must be at least 1.0")

    weights: List[float] = []
    span = cap_run - min_run
    for target in targets:
        longest = _max_run_length(target)
        if longest <= min_run:
            weights.append(1.0)
            continue
        if span <= 0:
            boost = max_boost
        else:
            effective = min(longest, cap_run)
            fraction = (effective - min_run) / span
            boost = 1.0 + (max_boost - 1.0) * float(fraction)
        weights.append(boost)
    if not weights:
        return torch.ones((0,), dtype=torch.float32)
    return torch.tensor(weights, dtype=torch.float32)


class CTCFastLoss(nn.Module):
    """Compute CTC loss directly from decoder logits."""

    def __init__(
        self,
        *,
        blank: int = BLANK_INDEX,
        reduction: str = "mean",
        zero_infinity: bool = True,
        homopolymer_weighting: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        if blank != BLANK_INDEX:
            raise ValueError("CTCFastLoss expects blank index 0 to align with decoder outputs")
        self.reduction = reduction
        self._blank = blank
        self._zero_infinity = zero_infinity
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self._ctc_per_sample: Optional[nn.CTCLoss] = None
        self._hpoly_cfg: Optional[Dict[str, object]] = None
        if homopolymer_weighting:
            enabled = homopolymer_weighting.get("enabled", True)
            if enabled:
                min_run = int(homopolymer_weighting.get("min_run", 4))
                cap_run = int(homopolymer_weighting.get("cap_run", max(min_run, 10)))
                max_boost = float(homopolymer_weighting.get("max_boost", 2.0))
                self._hpoly_cfg = {
                    "min_run": min_run,
                    "cap_run": cap_run,
                    "max_boost": max_boost,
                }
                self._ctc_per_sample = nn.CTCLoss(
                    blank=blank, reduction="none", zero_infinity=zero_infinity
                )
        self._logger = logging.getLogger("loss.ctc_fast")
        self._overflow_notified = False

    def _maybe_raise_length_overflow(
        self,
        logit_lengths: Tensor,
        target_lengths: Tensor,
        targets: List[Tensor],
        batch: Dict[str, Tensor],
    ) -> None:
        """Guard against silently clamping losses when targets exceed logits.

        PyTorch's ``CTCLoss`` returns ``inf`` (and subsequently ``0`` once
        ``zero_infinity=True`` is applied) whenever the input sequence is too
        short to realise the target, even if ``target_lengths <= logit_lengths``.
        This happens when the reference contains repeated bases that require
        interleaved blanks. We therefore compute the minimal viable CTC input
        length ``target_len + repeated_runs`` and compare that against the
        decoder budget, surfacing an actionable error instead of silently
        zeroing the loss.
        """

        if logit_lengths.numel() == 0 or target_lengths.numel() == 0:
            return

        minimal_lengths: List[int] = []
        for tgt_tensor, tgt_len in zip(targets, target_lengths.tolist()):
            minimal_lengths.append(minimal_ctc_input_length(tgt_tensor, tgt_len))

        minimal_tensor = torch.tensor(minimal_lengths, device=logit_lengths.device, dtype=torch.long)
        overflow = minimal_tensor > logit_lengths.to(dtype=torch.long, device=logit_lengths.device)
        if not overflow.any():
            return

        indices = torch.nonzero(overflow, as_tuple=False).view(-1).tolist()
        details: List[str] = []
        for idx in indices[:8]:
            required = int(minimal_tensor[idx].item())
            available = int(logit_lengths[idx].item())
            details.append(f"{idx}:required={required}>available={available}")

        segment_ids = batch.get("segment_id", [])
        if isinstance(segment_ids, list):
            segment_preview = segment_ids[:3]
            if len(segment_ids) > 3:
                segment_preview = segment_preview + [f"...(+{len(segment_ids) - 3})"]
        else:
            segment_preview = [segment_ids]

        message = (
            "CTCFastLoss detected decoder budgets that are insufficient for the reference. "
            "Increase fusion.num_queries / dynamic_query_cap or relax fast-path settings."
            f" offending_indices={details} segments={segment_preview}"
        )

        # Allow opting out in emergencies (e.g. quick exploratory runs) without code edits.
        if os.getenv("PF2_ALLOW_CTC_OVERFLOW", "0") == "1":
            if not self._overflow_notified:
                self._logger.warning("%s", message)
                self._overflow_notified = True
            return

        raise RuntimeError(message)

    @staticmethod
    def _normalize_padding_mask(
        mask: Tensor, target_steps: int, device: torch.device
    ) -> Tensor:
        if mask.dim() == 3:
            batch, reads, steps = mask.shape
            if steps != target_steps:
                if steps > target_steps:
                    mask = mask[..., :target_steps]
                else:
                    pad = torch.ones(
                        batch,
                        reads,
                        target_steps - steps,
                        dtype=mask.dtype,
                        device=mask.device,
                    )
                    mask = torch.cat((mask, pad), dim=-1)
            mask = mask.to(device=device, dtype=torch.bool)
            return mask.all(dim=1)
        if mask.dim() == 2:
            if mask.size(1) != target_steps:
                if mask.size(1) > target_steps:
                    mask = mask[:, :target_steps]
                else:
                    pad = torch.ones(
                        mask.size(0),
                        target_steps - mask.size(1),
                        dtype=mask.dtype,
                        device=mask.device,
                    )
                    mask = torch.cat((mask, pad), dim=1)
            return mask.to(device=device, dtype=torch.bool)
        raise ValueError("decoder_padding_mask must have shape (B, T) or (B, R, T)")

    @staticmethod
    def _gather_with_indices(
        tensor: Tensor, gather_indices: Tensor, lengths: Tensor
    ) -> Tensor:
        if gather_indices.numel() == 0:
            feature_dim = tensor.size(-1)
            return tensor.new_zeros((tensor.size(0), 0, feature_dim))

        batch, max_valid = gather_indices.shape
        feature_dim = tensor.size(-1)
        compact = tensor.new_zeros((batch, max_valid, feature_dim))
        for batch_idx in range(batch):
            valid_len = int(lengths[batch_idx].item()) if batch_idx < lengths.numel() else 0
            if valid_len <= 0:
                continue
            indices = gather_indices[batch_idx, :valid_len].to(dtype=torch.long, device=tensor.device)
            compact[batch_idx, :valid_len] = tensor[batch_idx].index_select(0, indices)
        return compact

    def _gather_with_mask(
        self, tensor: Tensor, mask: Tensor, lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if mask.numel() == 0:
            return tensor.new_zeros((tensor.size(0), 0, tensor.size(-1))), lengths

        valid_mask = ~mask
        inferred_lengths = valid_mask.sum(dim=1)
        max_valid = int(inferred_lengths.max().item()) if inferred_lengths.numel() > 0 else 0
        if max_valid == 0:
            feature_dim = tensor.size(-1)
            return tensor.new_zeros((tensor.size(0), 0, feature_dim)), inferred_lengths.to(lengths.device)

        feature_dim = tensor.size(-1)
        compact = tensor.new_zeros((tensor.size(0), max_valid, feature_dim))
        for batch_idx in range(tensor.size(0)):
            valid_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).view(-1)
            if valid_indices.numel() == 0:
                continue
            compact[batch_idx, : valid_indices.numel()] = tensor[batch_idx].index_select(0, valid_indices)
        return compact, inferred_lengths.to(device=lengths.device, dtype=lengths.dtype)

    def _resolve_logit_lengths(
        self, logits: Tensor, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, time_steps, _ = logits.shape

        logit_lengths = outputs.get("ctc_logit_lengths")
        if logit_lengths is not None:
            if logit_lengths.dim() != 1:
                raise ValueError("ctc_logit_lengths must be 1D")
            if logit_lengths.numel() != batch_size:
                raise ValueError("ctc_logit_lengths must match the batch dimension")
            return logit_lengths.to(dtype=torch.long, device=logits.device), None

        pad_mask = outputs.get("decoder_padding_mask")
        if pad_mask is None:
            pad_mask = batch.get("decoder_padding_mask")
        if pad_mask is not None:
            norm_mask = self._normalize_padding_mask(pad_mask, time_steps, logits.device)
            lengths = (~norm_mask).sum(dim=1)
            return lengths.to(dtype=torch.long, device=logits.device), norm_mask

        fallback = torch.full((batch_size,), time_steps, dtype=torch.long, device=logits.device)
        return fallback, None

    def _concat_targets(self, targets: List[Tensor], device: torch.device) -> Tensor:
        if not targets:
            return torch.zeros((0,), dtype=torch.long, device=device)
        pieces: List[Tensor] = []
        for tensor in targets:
            if tensor.numel() == 0:
                continue
            pieces.append(tensor.to(device=device, dtype=torch.long))
        if not pieces:
            return torch.zeros((0,), dtype=torch.long, device=device)
        return torch.cat(pieces, dim=0)

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        if "ctc_logits" not in outputs:
            raise KeyError("Decoder outputs must contain 'ctc_logits'")
        logits = outputs["ctc_logits"]
        if logits.dim() != 3:
            raise ValueError("ctc_logits must have shape (B, T, V)")
        batch_size, time_steps, vocab_size = logits.shape
        if vocab_size != EXPECTED_VOCAB:
            raise ValueError(
                f"ctc_logits must have last dimension {EXPECTED_VOCAB}, received {vocab_size}"
            )

        logit_lengths, padding_mask = self._resolve_logit_lengths(logits, outputs, batch)

        gather_indices = outputs.get("ctc_logit_gather_indices")
        max_valid = int(logit_lengths.max().item()) if logit_lengths.numel() > 0 else 0
        needs_compaction = logits.size(1) > max_valid
        if gather_indices is not None and needs_compaction:
            if gather_indices.dim() != 2:
                raise ValueError("ctc_logit_gather_indices must have shape (B, T)")
            if gather_indices.size(0) != batch_size:
                raise ValueError("ctc_logit_gather_indices must align with the batch dimension")
            logits = self._gather_with_indices(
                logits, gather_indices.to(device=logits.device, dtype=torch.long), logit_lengths
            )
        elif padding_mask is not None and needs_compaction:
            logits, inferred_lengths = self._gather_with_mask(logits, padding_mask, logit_lengths)
            logit_lengths = inferred_lengths.to(dtype=torch.long, device=logits.device)

        targets, target_lengths = _gather_reference(batch)
        if len(targets) != batch_size:
            raise ValueError("Gathered targets do not match batch size")

        device = logits.device
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        log_probs = torch.nan_to_num(log_probs, nan=float("-inf"), posinf=-1e4, neginf=-1e4)

        flat_targets = self._concat_targets(targets, device)
        target_lengths_cpu = target_lengths.to(dtype=torch.long, device="cpu")
        logit_lengths_cpu = logit_lengths.to(dtype=torch.long, device="cpu")

        self._maybe_raise_length_overflow(logit_lengths_cpu, target_lengths_cpu, targets, batch)

        if self._hpoly_cfg is None:
            return self.ctc(log_probs, flat_targets, logit_lengths_cpu, target_lengths_cpu)

        if self.reduction not in {"mean", "sum"}:
            raise ValueError("Homopolymer weighting requires reduction to be 'mean' or 'sum'")

        if self._ctc_per_sample is None:
            self._ctc_per_sample = nn.CTCLoss(
                blank=self._blank, reduction="none", zero_infinity=self._zero_infinity
            )

        loss_vec = self._ctc_per_sample(
            log_probs, flat_targets, logit_lengths_cpu, target_lengths_cpu
        )

        weights = _compute_hpoly_weights(
            targets,
            min_run=int(self._hpoly_cfg["min_run"]),
            cap_run=int(self._hpoly_cfg["cap_run"]),
            max_boost=float(self._hpoly_cfg["max_boost"]),
        ).to(device=loss_vec.device, dtype=loss_vec.dtype)

        if weights.numel() == 0:
            return loss_vec.sum() * 0.0

        if self.reduction == "sum":
            return (loss_vec * weights).sum()

        target_lengths_f = target_lengths_cpu.to(
            device=loss_vec.device, dtype=loss_vec.dtype
        ).clamp_min(1)
        per_sample = loss_vec / target_lengths_f
        weighted = per_sample * weights
        return weighted.sum() / weights.sum().clamp_min(1e-6)


@LOSS_REGISTRY.register("ctc_fast")
def build_ctc_fast_loss(**kwargs: object) -> nn.Module:
    return CTCFastLoss(**kwargs)


__all__ = ["CTCFastLoss", "build_ctc_fast_loss", "minimal_ctc_input_length"]