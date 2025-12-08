"""Transformer decoder head producing logits for standard CTC training."""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .base import DECODER_REGISTRY, DecoderBase
from .ctc_crf import TemporalTransformer

NUM_BASES = 4
BLANK_INDEX = 0
VOCAB_SIZE = NUM_BASES + 1

LOGGER = logging.getLogger("decoder")


class CTCDecoder(DecoderBase):
    """Decoder that emits blank+base logits for vanilla CTC objectives."""

    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        use_temporal_positional_encoding: bool = False,
        enable_duration_head: bool = False,
        duration_head_hidden_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if ffn_multiplier < 1.0:
            raise ValueError("ffn_multiplier must be >= 1.0")
        if duration_head_hidden_scale <= 0:
            raise ValueError("duration_head_hidden_scale must be positive")

        self.model_dim = model_dim
        self.enable_duration_head = bool(enable_duration_head)
        self.transformer = TemporalTransformer(
            hidden_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            use_positional_encoding=use_temporal_positional_encoding,
        )
        self.output = nn.Linear(model_dim, VOCAB_SIZE)
        if self.enable_duration_head:
            hidden_dim = int(round(model_dim * float(duration_head_hidden_scale)))
            hidden_dim = max(hidden_dim, model_dim)
            self.duration_head = nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.duration_head = None

    def _validate_inputs(self, batch: Dict[str, Tensor]) -> Tensor:
        if "embedding" not in batch:
            raise KeyError("Decoder expects 'embedding' in batch")
        hidden = batch["embedding"]
        if hidden.dim() != 3:
            raise ValueError("Decoder expects embedding tensor of shape (B, T, D)")
        if hidden.size(-1) != self.model_dim:
            raise ValueError(
                f"Decoder configured for model_dim={self.model_dim}, received {hidden.size(-1)}"
            )
        if not torch.isfinite(hidden).all():
            bad = ~torch.isfinite(hidden)
            bad_batches = bad.any(dim=(1, 2))
            indices = torch.nonzero(bad_batches, as_tuple=False).view(-1)
            for idx in indices.tolist()[:8]:
                row = hidden[idx]
                LOGGER.warning(
                    "decoder input hidden non-finite | b=%d | any_finite=%s | min=%.3e max=%.3e",
                    idx,
                    torch.isfinite(row).any().item(),
                    torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).min().item(),
                    torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).max().item(),
                )
        return hidden

    def _resolve_lengths(
        self,
        hidden: Tensor,
        decoder_padding_mask: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Tensor]:
        key_padding_mask: Optional[Tensor] = None
        if decoder_padding_mask is not None:
            if decoder_padding_mask.dim() == 3:
                batch, reads, steps = decoder_padding_mask.shape
                if batch != hidden.size(0):
                    raise ValueError(
                        "decoder_padding_mask batch dimension must match embedding batch dimension"
                    )
                mask = decoder_padding_mask
                if steps != hidden.size(1):
                    if steps > hidden.size(1):
                        mask = mask[..., : hidden.size(1)]
                    else:
                        pad = torch.ones(
                            batch,
                            reads,
                            hidden.size(1) - steps,
                            dtype=mask.dtype,
                            device=mask.device,
                        )
                        mask = torch.cat((mask, pad), dim=-1)
                mask = mask.to(device=hidden.device, dtype=torch.bool)
                key_padding_mask = mask.all(dim=1)
            elif decoder_padding_mask.dim() == 2:
                if decoder_padding_mask.size(0) != hidden.size(0) or decoder_padding_mask.size(1) != hidden.size(1):
                    raise ValueError("decoder_padding_mask must align with embedding shape")
                key_padding_mask = decoder_padding_mask.to(device=hidden.device, dtype=torch.bool)
            else:
                raise ValueError("decoder_padding_mask must have shape (B, T) or (B, R, T)")
            lengths = (~key_padding_mask).sum(dim=1)
        else:
            lengths = torch.full(
                (hidden.size(0),), hidden.size(1), dtype=torch.long, device=hidden.device
            )
        return key_padding_mask, lengths

    def _build_gather_indices(
        self, lengths: Tensor, key_padding_mask: Optional[Tensor], total_steps: int
    ) -> Tensor:
        if lengths.numel() == 0:
            return torch.zeros((0, 0), dtype=torch.long, device=lengths.device)

        max_valid = int(lengths.max().item())
        if max_valid <= 0:
            return torch.zeros((lengths.size(0), 0), dtype=torch.long, device=lengths.device)

        if key_padding_mask is None:
            base = torch.arange(total_steps, device=lengths.device, dtype=torch.long)
            return base[:max_valid].expand(lengths.size(0), -1).contiguous()

        gather = torch.zeros((lengths.size(0), max_valid), dtype=torch.long, device=lengths.device)
        for batch_idx in range(lengths.size(0)):
            valid_len = int(lengths[batch_idx].item())
            if valid_len <= 0:
                continue
            valid_indices = torch.nonzero(~key_padding_mask[batch_idx], as_tuple=False).view(-1)
            if valid_indices.numel() < valid_len:
                raise RuntimeError("decoder padding mask reported fewer valid entries than lengths")
            gather[batch_idx, :valid_len] = valid_indices[:valid_len]
        return gather

    @staticmethod
    def _compact_time_dimension(tensor: Tensor, gather_indices: Tensor, lengths: Tensor) -> Tensor:
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
            indices = gather_indices[batch_idx, :valid_len]
            compact[batch_idx, :valid_len] = tensor[batch_idx].index_select(0, indices)
        return compact

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden = self._validate_inputs(batch)
        pad_mask = batch.get("decoder_padding_mask")
        key_padding_mask, lengths = self._resolve_lengths(hidden, pad_mask)

        encoded = self.transformer(hidden, key_padding_mask=key_padding_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1e4, neginf=-1e4)

        gather_indices = self._build_gather_indices(lengths, key_padding_mask, hidden.size(1))
        compact_encoded = self._compact_time_dimension(encoded, gather_indices, lengths)

        logits = self.output(compact_encoded)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        output = dict(batch)
        output["embedding"] = encoded
        output["ctc_logits"] = logits
        output["ctc_logit_lengths"] = lengths
        output["ctc_logit_gather_indices"] = gather_indices
        if self.enable_duration_head and self.duration_head is not None:
            duration_logits = self.duration_head(compact_encoded).squeeze(-1)
            duration_logits = torch.nan_to_num(
                duration_logits, nan=0.0, posinf=1e4, neginf=-1e4
            )
            output["duration_logits"] = duration_logits
            if "duration_target" in batch:
                output.setdefault("duration_target", batch["duration_target"])
        return output


@DECODER_REGISTRY.register("ctc")
def build_ctc_decoder(**kwargs: object) -> DecoderBase:
    return CTCDecoder(**kwargs)


__all__ = ["CTCDecoder", "build_ctc_decoder"]