"""Recurrent neural network transducer (RNN-T) decoder implementation."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .base import DECODER_REGISTRY, DecoderBase

NUM_BASES = 4
BLANK_INDEX = 0
VOCAB_SIZE = NUM_BASES + 1

LOGGER = logging.getLogger("decoder")


class _PredictionNetwork(nn.Module):
    """Simple embedding + GRU prediction network."""

    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=BLANK_INDEX)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, targets: Tensor, lengths: Tensor) -> Tensor:
        if targets.dim() != 2:
            raise ValueError("Prediction network expects targets with shape (B, U)")

        embedded = self.dropout(self.embedding(targets))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return unpacked


class RNNTDecoder(DecoderBase):
    """Decoder that produces logits compatible with RNN-T objectives."""

    def __init__(
        self,
        *,
        encoder_dim: int = 256,
        prediction_dim: int = 256,
        joint_dim: int = 320,
        prediction_dropout: float = 0.1,
        prediction_layers: int = 1,
    ) -> None:
        super().__init__()
        if encoder_dim <= 0:
            raise ValueError("encoder_dim must be positive")
        if prediction_dim <= 0:
            raise ValueError("prediction_dim must be positive")
        if joint_dim <= 0:
            raise ValueError("joint_dim must be positive")
        if prediction_layers < 1:
            raise ValueError("prediction_layers must be >= 1")

        self.encoder_dim = encoder_dim
        self.prediction_dim = prediction_dim
        self.joint_dim = joint_dim

        self.prediction = _PredictionNetwork(
            vocab_size=VOCAB_SIZE,
            hidden_dim=prediction_dim,
            dropout=prediction_dropout,
            num_layers=prediction_layers,
        )
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.prediction_proj = nn.Linear(prediction_dim, joint_dim)
        self.joint = nn.Sequential(nn.Tanh(), nn.Linear(joint_dim, VOCAB_SIZE))

    def _prediction_step(
        self, token: Tensor, state: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Run a single prediction-network step for a provided token."""

        embedded = self.prediction.dropout(self.prediction.embedding(token))
        output, new_state = self.prediction.rnn(embedded, state)
        return output[:, -1, :], new_state

    def _validate_inputs(self, batch: Dict[str, Tensor]) -> Tensor:
        if "embedding" not in batch:
            raise KeyError("Decoder expects 'embedding' in batch")
        hidden = batch["embedding"]
        if hidden.dim() != 3:
            raise ValueError("Decoder expects embedding tensor of shape (B, T, D)")
        if hidden.size(-1) != self.encoder_dim:
            raise ValueError(
                f"Decoder configured for encoder_dim={self.encoder_dim}, received {hidden.size(-1)}"
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

    @staticmethod
    def _resolve_lengths(hidden: Tensor, decoder_padding_mask: Optional[Tensor]) -> Tensor:
        if decoder_padding_mask is None:
            return torch.full((hidden.size(0),), hidden.size(1), device=hidden.device, dtype=torch.long)

        mask = decoder_padding_mask
        if mask.dim() == 3:
            mask = mask.all(dim=1)
        if mask.dim() != 2:
            raise ValueError("decoder_padding_mask must have shape (B, T) or (B, R, T)")
        if mask.size(0) != hidden.size(0):
            raise ValueError("decoder_padding_mask batch dimension must match embedding batch dimension")
        if mask.size(1) != hidden.size(1):
            if mask.size(1) > hidden.size(1):
                mask = mask[:, : hidden.size(1)]
            else:
                pad = torch.ones(
                    mask.size(0), hidden.size(1) - mask.size(1), dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat((mask, pad), dim=1)
        return (~mask.to(device=hidden.device, dtype=torch.bool)).sum(dim=1)

    @staticmethod
    def _gather_targets(batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        reference_index = batch.get("reference_index")
        reference_lengths = batch.get("reference_lengths")

        if not isinstance(reference_index, Tensor) or not isinstance(reference_lengths, Tensor):
            raise KeyError(
                "RNNT decoder expects 'reference_index' and 'reference_lengths' from FASTA references"
            )

        reference_index = reference_index.long()
        reference_lengths = reference_lengths.to(dtype=torch.long)

        if reference_index.dim() == 1:
            reference_index = reference_index.unsqueeze(0)
        if reference_index.dim() != 2:
            raise ValueError("'reference_index' must have shape (batch, time)")

        if reference_lengths.dim() == 0:
            reference_lengths = reference_lengths.unsqueeze(0)
        if reference_lengths.dim() != 1:
            raise ValueError("'reference_lengths' must have shape (batch,)")
        if reference_lengths.numel() != reference_index.size(0):
            raise ValueError("reference_lengths must align with the batch dimension")

        max_len = int(reference_lengths.max().item()) if reference_lengths.numel() > 0 else 0
        targets = reference_index.new_full((reference_index.size(0), max_len), BLANK_INDEX)
        for idx in range(reference_index.size(0)):
            length = int(reference_lengths[idx].item())
            if length <= 0:
                raise ValueError("reference_lengths must be positive for RNNT decoding")
            if length > reference_index.size(1):
                raise ValueError("reference_length exceeds available reference_index entries")
            targets[idx, :length] = reference_index[idx, :length]
        return targets, reference_lengths

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden = self._validate_inputs(batch)
        input_lengths = self._resolve_lengths(hidden, batch.get("decoder_padding_mask"))

        targets, target_lengths = self._gather_targets(batch)
        device = hidden.device
        targets = targets.to(device=device, dtype=torch.long)
        target_lengths = target_lengths.to(device=device, dtype=torch.long)

        predictor_inputs = targets.new_full((targets.size(0), targets.size(1) + 1), BLANK_INDEX)
        predictor_inputs[:, 1:] = targets
        predictor_lengths = target_lengths + 1

        prediction = self.prediction(predictor_inputs, predictor_lengths)
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=1e4, neginf=-1e4)

        enc_proj = self.encoder_proj(hidden)
        pred_proj = self.prediction_proj(prediction)

        joint = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)
        joint = self.joint(joint)
        joint = torch.nan_to_num(joint, nan=0.0, posinf=1e4, neginf=-1e4)

        output = dict(batch)
        output["rnnt_logits"] = joint
        output["rnnt_logit_lengths"] = input_lengths
        output["rnnt_target_lengths"] = predictor_lengths
        output["decoder_padding_mask"] = batch.get("decoder_padding_mask")
        return output

    def greedy_decode(
        self,
        hidden: Tensor,
        *,
        input_lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        max_symbols_per_step: int = 32,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        """Greedily decode encoder representations with the RNN-T predictor.

        Args:
            hidden: Encoder representations of shape ``(B, T, D)``.
            input_lengths: Optional tensor of per-sample time steps to process.
            padding_mask: Optional mask indicating padded encoder steps.
            max_symbols_per_step: Safety bound on the number of symbols emitted per
                encoder frame to avoid infinite loops when no blanks are produced.

        Returns:
            A tuple of decoded token tuples and a tensor of per-sample encoder lengths
            used during decoding.
        """

        if max_symbols_per_step <= 0:
            raise ValueError("max_symbols_per_step must be positive for RNNT decoding")

        hidden = self._validate_inputs({"embedding": hidden})
        lengths = input_lengths
        if lengths is None:
            lengths = self._resolve_lengths(hidden, padding_mask)
        if lengths.dim() != 1 or lengths.numel() != hidden.size(0):
            raise ValueError("input_lengths must be 1D and align with batch size")

        device = hidden.device
        decoded: Tuple[Tuple[int, ...], ...] = tuple()
        blank_token = torch.tensor([[BLANK_INDEX]], device=device, dtype=torch.long)

        for batch_idx in range(hidden.size(0)):
            time_len = int(lengths[batch_idx].item()) if batch_idx < lengths.numel() else hidden.size(1)
            if time_len <= 0:
                decoded += (tuple(),)
                continue

            enc_seq = hidden[batch_idx, :time_len]
            pred_vec, pred_state = self._prediction_step(blank_token, None)

            tokens: Tuple[int, ...] = tuple()
            for t in range(time_len):
                symbols = 0
                enc_proj = self.encoder_proj(enc_seq[t : t + 1])
                while symbols < max_symbols_per_step:
                    joint_input = enc_proj + self.prediction_proj(pred_vec)
                    logits = self.joint(joint_input)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                    next_token = int(logits.argmax(dim=-1).item())
                    if next_token == BLANK_INDEX:
                        break

                    tokens += (next_token,)
                    symbols += 1
                    token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
                    pred_vec, pred_state = self._prediction_step(token_tensor, pred_state)

                if symbols >= max_symbols_per_step:
                    LOGGER.warning(
                        "RNNT greedy decode hit max_symbols_per_step=%d at batch=%d time=%d",
                        max_symbols_per_step,
                        batch_idx,
                        t,
                    )

            decoded += (tokens,)

        return decoded, lengths.to(device=device, dtype=torch.long)

    def beam_decode(
        self,
        hidden: Tensor,
        *,
        input_lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        beam_width: int = 4,
        max_symbols_per_step: int = 32,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        """Beam search decoding over encoder representations.

        Args:
            hidden: Encoder representations of shape ``(B, T, D)``.
            input_lengths: Optional tensor of per-sample time steps to process.
            padding_mask: Optional mask indicating padded encoder steps.
            beam_width: Number of candidate sequences to keep after each frame.
            max_symbols_per_step: Maximum non-blank emissions per frame.

        Returns:
            Tuple of decoded token tuples (best hypothesis per sample) and tensor of
            per-sample encoder lengths consumed during decoding.
        """

        if beam_width <= 0:
            raise ValueError("beam_width must be positive for RNNT beam decoding")
        if max_symbols_per_step <= 0:
            raise ValueError("max_symbols_per_step must be positive for RNNT decoding")

        hidden = self._validate_inputs({"embedding": hidden})
        lengths = input_lengths
        if lengths is None:
            lengths = self._resolve_lengths(hidden, padding_mask)
        if lengths.dim() != 1 or lengths.numel() != hidden.size(0):
            raise ValueError("input_lengths must be 1D and align with batch size")

        device = hidden.device
        decoded: Tuple[Tuple[int, ...], ...] = tuple()
        blank_token = torch.tensor([[BLANK_INDEX]], device=device, dtype=torch.long)

        for batch_idx in range(hidden.size(0)):
            time_len = int(lengths[batch_idx].item()) if batch_idx < lengths.numel() else hidden.size(1)
            if time_len <= 0:
                decoded += (tuple(),)
                continue

            enc_seq = hidden[batch_idx, :time_len]
            pred_vec, pred_state = self._prediction_step(blank_token, None)

            beam = [
                (tuple(), 0.0, pred_state, pred_vec, 0),
            ]  # (tokens, score, state, pred_vec, emitted_symbols)

            for t in range(time_len):
                enc_proj = self.encoder_proj(enc_seq[t : t + 1])
                next_frame: list[Tuple[Tuple[int, ...], float, Optional[Tensor], Tensor, int]] = []

                queue = list(beam)
                while queue:
                    tokens, score, state, pvec, emitted = queue.pop()
                    joint_input = enc_proj + self.prediction_proj(pvec)
                    logits = self.joint(joint_input)
                    log_probs = torch.log_softmax(
                        torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1
                    )
                    log_probs = log_probs.view(-1)

                    blank_logp = float(log_probs[BLANK_INDEX].item())
                    next_frame.append((tokens, score + blank_logp, state, pvec, emitted))

                    if emitted >= max_symbols_per_step:
                        continue

                    non_blank_topk = torch.topk(log_probs[1:], k=min(beam_width, VOCAB_SIZE - 1))
                    for offset, logp in zip(non_blank_topk.indices, non_blank_topk.values):
                        token_id = int(offset.item() + 1)
                        token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
                        new_vec, new_state = self._prediction_step(token_tensor, state)
                        queue.append((tokens + (token_id,), score + float(logp.item()), new_state, new_vec, emitted + 1))

                next_frame.sort(key=lambda item: item[1], reverse=True)
                beam = [
                    (tokens, score, state, pvec, 0)
                    for tokens, score, state, pvec, _ in next_frame[:beam_width]
                ]

            best_tokens = beam[0][0] if beam else tuple()
            decoded += (best_tokens,)

        return decoded, lengths.to(device=device, dtype=torch.long)


@DECODER_REGISTRY.register("rnnt")
def build_rnnt_decoder(**kwargs: object) -> DecoderBase:
    return RNNTDecoder(**kwargs)


__all__ = ["RNNTDecoder", "build_rnnt_decoder"]