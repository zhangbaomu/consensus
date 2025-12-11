"""Transformer-based decoder with CTC-CRF emissions and decoding."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .base import DECODER_REGISTRY, DecoderBase
from ...utils.k2_ctc_crf import (
    BLANK_LABEL,
    NUM_BLANK_CLASSES,
    build_blank_dense_inputs,
    build_blank_denominator_graph,
    build_ctc_blank_target_graph,
    build_dense_inputs,
    build_transition_graph,
    create_fsa_vec,
)


def _require_k2() -> ModuleType:
    try:
        import k2  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard for optional dependency
        raise ImportError(
            "The 'k2' package is required for the CTC-CRF decoder. Install it via 'pip install k2'."
        ) from exc
    return k2  # type: ignore


BASE_VOCAB = ("A", "C", "G", "T")
NUM_BASES = len(BASE_VOCAB)
STATE_ENTER_OFFSET = 0
STATE_EXTEND_OFFSET = NUM_BASES
NUM_STATES = NUM_BASES * 2


def _build_alibi_slopes(num_heads: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return ALiBi slopes following the original paper's heuristic."""

    def _get_slopes(n: int) -> List[float]:
        if n == 1:
            return [1.0]
        power_of_two = 2 ** int(torch.log2(torch.tensor(float(n))).item())
        base = 2.0 ** (-8.0 / power_of_two)
        slopes = [base ** i for i in range(power_of_two)]
        if power_of_two < n:
            extra = _get_slopes(2 * power_of_two)
            slopes.extend(extra[::2][: n - power_of_two])
        return slopes

    return torch.tensor(_get_slopes(num_heads), device=device, dtype=dtype)


def _build_alibi_bias(sequence_length: int, num_heads: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Construct ALiBi bias matrix using normalized progress indices."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    slopes = _build_alibi_slopes(num_heads, device=device, dtype=dtype).view(num_heads, 1, 1)
    if sequence_length == 1:
        progress = torch.zeros(sequence_length, device=device, dtype=dtype)
    else:
        progress = torch.linspace(0.0, 1.0, steps=sequence_length, device=device, dtype=dtype)
    diff = progress.view(1, sequence_length, 1) - progress.view(1, 1, sequence_length)
    return slopes * diff


class TemporalMultiHeadAttention(nn.Module):
    """Multi-head attention with optional additive bias per head."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor: Tensor) -> Tensor:
        batch, seq_len, _ = tensor.shape
        return (
            tensor.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: Tensor,
        *,
        attention_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            attn = attn + attention_bias.unsqueeze(0)

        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(expanded_mask, float("-inf"))

            keys_all_masked = key_padding_mask.all(dim=1)
            if keys_all_masked.any():
                bad = keys_all_masked.nonzero(as_tuple=False).view(-1)
                safe_attn = attn.clone()
                safe_attn[bad] = 0.0
                weights = torch.softmax(safe_attn, dim=-1)
                weights[bad] = 0.0
            else:
                weights = torch.softmax(attn, dim=-1)
        else:
            weights = torch.softmax(attn, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.hidden_dim)
        return self.out_proj(context)


class TemporalEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_multiplier: float) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim * ffn_multiplier)
        if ffn_dim < hidden_dim:
            raise ValueError("ffn_multiplier must be >= 1.0")

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = TemporalMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        *,
        attention_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        residual = x
        attn_input = self.norm1(x)
        attn_output = self.attn(attn_input, attention_bias=attention_bias, key_padding_mask=key_padding_mask)
        x = residual + self.drop_path(attn_output)

        residual = x
        ff_input = self.norm2(x)
        x = residual + self.ffn(ff_input)
        return x


class TemporalTransformer(nn.Module):
    """Stack of temporal encoder layers with shared ALiBi bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList(
            [TemporalEncoderLayer(hidden_dim, num_heads, dropout, ffn_multiplier) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding

    def _build_positional_encoding(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if seq_len <= 0:
            return torch.zeros(seq_len, self.hidden_dim, device=device, dtype=dtype)

        positions = torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=dtype)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(self.hidden_dim, 1))
        )
        encoding = torch.zeros(seq_len, self.hidden_dim, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        cos_components = torch.cos(positions.unsqueeze(-1) * div_term)
        if self.hidden_dim % 2 == 0:
            encoding[:, 1::2] = cos_components
        else:
            encoding[:, 1::2] = cos_components[:, : encoding[:, 1::2].shape[1]]
        return encoding

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape
        if seq_len == 0:
            return x

        device = x.device
        dtype = x.dtype
        attention_bias = _build_alibi_bias(seq_len, self.num_heads, device=device, dtype=dtype)

        if self.use_positional_encoding:
            positional = self._build_positional_encoding(seq_len, device=device, dtype=dtype)
            out = x + positional.unsqueeze(0)
        else:
            out = x
        for layer in self.layers:
            out = layer(out, attention_bias=attention_bias, key_padding_mask=key_padding_mask)
        return self.final_norm(out)


def _legal_transition_indices() -> Tensor:
    indices: List[Tuple[int, int]] = []
    for base in range(NUM_BASES):
        enter = STATE_ENTER_OFFSET + base
        extend = STATE_EXTEND_OFFSET + base
        indices.append((enter, extend))  # enter -> extend
        indices.append((extend, extend))  # extend -> extend
        for next_base in range(NUM_BASES):
            indices.append((enter, STATE_ENTER_OFFSET + next_base))  # enter -> enter(next)
        for next_base in range(NUM_BASES):
            indices.append((extend, STATE_ENTER_OFFSET + next_base))  # extend -> enter(next)
    return torch.tensor(indices, dtype=torch.long)


@dataclass
class DecoderOutputs:
    emissions: Tensor
    transition: Tensor
    bos: Tensor
    eos: Tensor
    viterbi_sequences: Optional[List[List[int]]]
    viterbi_run_lengths: Optional[List[List[int]]]


class CTCCRFDecoder(DecoderBase):
    """Decoder producing enter/extend emissions for a CTC-CRF objective."""

    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        *,
        return_viterbi: bool = False,
        clamp_transition: float = 50.0,
        use_temporal_positional_encoding: bool = False,
        viterbi_search_beam: float = 20.0,
        viterbi_output_beam: float = 8.0,
        viterbi_min_active_states: int = 1,
        viterbi_max_active_states: int = 4096,
        variant: str = "ctc_crf_enter_extend",
        blank_num_classes: int = NUM_BLANK_CLASSES,
        learn_transition_bias: bool = True,
        denominator: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        self.model_dim = model_dim
        self.return_viterbi = return_viterbi
        self.clamp_transition = clamp_transition
        self.viterbi_search_beam = float(viterbi_search_beam)
        self.viterbi_output_beam = float(viterbi_output_beam)
        self.viterbi_min_active_states = int(viterbi_min_active_states)
        self.viterbi_max_active_states = int(viterbi_max_active_states)
        normalized_variant = variant.lower()
        if normalized_variant in {"enter_extend", "ctc_crf_enter_extend"}:
            self.variant = "ctc_crf_enter_extend"
        elif normalized_variant in {"blank", "ctc_crf_blank"}:
            self.variant = "ctc_crf_blank"
        else:
            raise ValueError(
                "variant must be either 'ctc_crf_enter_extend' or 'ctc_crf_blank'"
            )
        self.learn_transition_bias = bool(learn_transition_bias)

        self.transformer = TemporalTransformer(
            hidden_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            use_positional_encoding=use_temporal_positional_encoding,
        )
        if self.variant == "ctc_crf_enter_extend":
            self.enter_head = nn.Linear(model_dim, NUM_BASES)
            self.extend_head = nn.Linear(model_dim, NUM_BASES)

            indices = _legal_transition_indices()
            self.register_buffer("transition_indices", indices, persistent=False)
            self.transition_scores = nn.Parameter(torch.zeros(indices.size(0)))
            self.bos_scores = nn.Parameter(torch.zeros(NUM_STATES))
            self.eos_scores = nn.Parameter(torch.zeros(NUM_STATES))
            self.blank_head = None
            self.blank_num_classes = blank_num_classes
        else:
            if blank_num_classes != NUM_BLANK_CLASSES:
                raise ValueError(
                    f"blank_num_classes must be {NUM_BLANK_CLASSES} (blank + {NUM_BASES} bases)"
                )
            self.enter_head = None
            self.extend_head = None
            self.blank_num_classes = blank_num_classes
            self.blank_head = nn.Linear(model_dim, blank_num_classes)
            # Alias the emission head under the plain CTC naming scheme so that
            # checkpoints trained with the vanilla decoder (which exposes
            # ``decoder.output.*`` parameters) line up with this variant.
            self.output = self.blank_head
            self._initialize_blank_denominator(denominator or {}, learn_transition_bias)

    def _initialize_blank_denominator(
        self, config: Dict[str, object], learnable: bool
    ) -> None:
        denom_type = str(config.get("type", "ctc_grammar")).lower()
        if denom_type not in {"ctc_grammar", "bigram"}:
            raise ValueError(
                "denominator.type must be 'ctc_grammar' or 'bigram' for blank variant"
            )
        self.blank_denominator_type = denom_type
        self.blank_allow_same_repeat = bool(config.get("allow_same_base_repeat", True))

        if denom_type == "ctc_grammar":
            self._register_blank_bias(
                "blank_bias_blank_blank",
                torch.tensor(0.0, dtype=torch.float32),
                learnable,
            )
            self._register_blank_bias(
                "blank_bias_blank_to_base",
                torch.tensor(0.0, dtype=torch.float32),
                learnable,
            )
            self._register_blank_bias(
                "blank_bias_base_to_blank",
                torch.tensor(0.0, dtype=torch.float32),
                learnable,
            )
            self._register_blank_bias(
                "blank_bias_base_to_other",
                torch.tensor(0.0, dtype=torch.float32),
                learnable,
            )
            self._register_blank_bias(
                "blank_bias_same_repeat",
                torch.tensor(0.0, dtype=torch.float32),
                learnable,
            )
        else:
            init_tensor = torch.zeros(self.blank_num_classes, self.blank_num_classes)
            self._register_blank_bias("blank_transition_matrix", init_tensor, learnable)

    def _register_blank_bias(
        self, name: str, value: torch.Tensor | float, learnable: bool
    ) -> None:
        template = torch.as_tensor(value, dtype=torch.float32)
        tensor = torch.zeros_like(template)
        if learnable:
            param = nn.Parameter(tensor)
            setattr(self, name, param)
        else:
            self.register_buffer(name, tensor, persistent=False)

    def _blank_bias_value(self, name: str) -> torch.Tensor:
        bias = getattr(self, name)
        if isinstance(bias, torch.Tensor):
            return bias
        raise AttributeError(f"Blank transition bias '{name}' is not defined")

    def _blank_transition_matrix(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        clamp_val = float(self.clamp_transition)
        if self.blank_denominator_type == "ctc_grammar":
            blank_blank = torch.clamp(
                self._blank_bias_value("blank_bias_blank_blank"),
                min=-clamp_val,
                max=clamp_val,
            ).to(device=device, dtype=dtype)
            blank_to_base = torch.clamp(
                self._blank_bias_value("blank_bias_blank_to_base"),
                min=-clamp_val,
                max=clamp_val,
            ).to(device=device, dtype=dtype)
            base_to_blank = torch.clamp(
                self._blank_bias_value("blank_bias_base_to_blank"),
                min=-clamp_val,
                max=clamp_val,
            ).to(device=device, dtype=dtype)
            base_to_other = torch.clamp(
                self._blank_bias_value("blank_bias_base_to_other"),
                min=-clamp_val,
                max=clamp_val,
            ).to(device=device, dtype=dtype)
            same_repeat = torch.clamp(
                self._blank_bias_value("blank_bias_same_repeat"),
                min=-clamp_val,
                max=clamp_val,
            ).to(device=device, dtype=dtype)

            num_bases = self.blank_num_classes - 1
            if num_bases <= 0:
                matrix = blank_blank.reshape(1, 1)
            else:
                blank_to_base_vec = blank_to_base.view(1).expand(num_bases)
                blank_row = torch.cat(
                    [blank_blank.unsqueeze(0), blank_to_base_vec], dim=0
                )

                base_to_other_rows = base_to_other.unsqueeze(0).expand(num_bases, num_bases)
                diag_mask = torch.eye(num_bases, device=device, dtype=torch.bool)
                if self.blank_allow_same_repeat:
                    same_repeat_matrix = same_repeat.unsqueeze(0).expand(
                        num_bases, num_bases
                    )
                    base_body = torch.where(
                        diag_mask, same_repeat_matrix, base_to_other_rows
                    )
                else:
                    neg_inf = torch.full((), float("-inf"), device=device, dtype=dtype)
                    base_body = torch.where(diag_mask, neg_inf, base_to_other_rows)

                base_to_blank_col = base_to_blank.view(1).expand(num_bases).unsqueeze(1)
                base_rows = torch.cat([base_to_blank_col, base_body], dim=1)

                matrix = torch.cat([blank_row.unsqueeze(0), base_rows], dim=0)
        else:
            matrix = torch.clamp(
                self._blank_bias_value("blank_transition_matrix"),
                min=-clamp_val,
                max=clamp_val,
            )
            if matrix.shape != (self.blank_num_classes, self.blank_num_classes):
                raise ValueError(
                    "blank transition matrix must have shape (num_classes, num_classes)"
                )
        return matrix.to(device=device, dtype=dtype)

    def _build_transition_matrix(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        rows = self.transition_indices[:, 0]
        cols = self.transition_indices[:, 1]
        values = self.transition_scores.clamp(min=-self.clamp_transition, max=self.clamp_transition)

        flat = torch.full((NUM_STATES * NUM_STATES,), float("-inf"), device=device, dtype=dtype)
        indices = rows * NUM_STATES + cols
        full = flat.scatter(0, indices.to(device=device), values.to(device=device, dtype=dtype))
        return full.view(NUM_STATES, NUM_STATES)

    def _compute_emissions(self, hidden: Tensor) -> Tensor:
        if not torch.isfinite(hidden).all():
            logger = logging.getLogger("decoder")
            with torch.no_grad():
                detached = hidden.detach()
                bad = ~torch.isfinite(detached)
                bad_batch = bad.any(dim=(1, 2))
                if bad_batch.any():
                    idxs = (
                        torch.nonzero(bad_batch, as_tuple=False)
                        .view(-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    safe_hidden = torch.nan_to_num(detached, nan=0.0, posinf=0.0, neginf=0.0)
                    for batch_idx in idxs[:8]:
                        row = safe_hidden[batch_idx]
                        logger.warning(
                            "decoder hidden non-finite | b=%d | any_finite=%s | min=%.3e max=%.3e",
                            batch_idx,
                            bool(torch.isfinite(detached[batch_idx]).any().item()),
                            float(row.min().item()),
                            float(row.max().item()),
                        )

        enter = self.enter_head(hidden)
        extend = self.extend_head(hidden)
        return torch.cat([enter, extend], dim=-1)

    @staticmethod
    def _blank_label_to_char(label: int) -> str:
        if label == BLANK_LABEL:
            return "-"
        base_index = label - 1
        if 1 <= base_index <= NUM_BASES:
            return BASE_VOCAB[base_index - 1]
        return "?"

    def _forward_blank_variant(
        self,
        batch: Dict[str, Tensor],
        encoded: Tensor,
        key_padding_mask: Optional[Tensor],
        lengths: Tensor,
    ) -> Dict[str, Tensor]:
        logits = self.blank_head(encoded)
        logits = torch.log_softmax(logits, dim=-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if key_padding_mask is not None:
            mask = key_padding_mask.to(dtype=torch.bool)
            logits = logits.masked_fill(mask.unsqueeze(-1), float("-inf"))

        transition = self._blank_transition_matrix(
            device=logits.device, dtype=torch.float32
        )

        output = dict(batch)
        output["embedding"] = encoded
        output["ctc_variant"] = "ctc_crf_blank"
        output["ctc_blank_logits"] = logits
        output["ctc_blank_transition"] = transition
        output["decoder_lengths"] = lengths

        if key_padding_mask is not None:
            output.setdefault("decoder_padding_mask", key_padding_mask)

        if self.return_viterbi:
            (
                sequences,
                collapsed_tokens,
                char_run_lengths,
                frame_run_lengths,
                collapsed_texts,
                rle_chars,
                rle_frames,
            ) = self._viterbi_blank(logits, transition, lengths)
            output["viterbi_sequence"] = sequences
            output["viterbi_collapsed_tokens"] = collapsed_tokens
            output["viterbi_char_run_length"] = char_run_lengths
            output["viterbi_frame_run_length"] = frame_run_lengths
            output["viterbi_run_length"] = frame_run_lengths
            output["viterbi_collapsed_text"] = collapsed_texts
            output["viterbi_rle_chars"] = rle_chars
            output["viterbi_rle_frames"] = rle_frames

        return output

    def _viterbi_blank(
        self,
        logits: Tensor,
        transition: Tensor,
        lengths: Tensor,
    ) -> Tuple[
        List[List[int]],
        List[List[int]],
        List[List[int]],
        List[List[int]],
        List[str],
        List[List[Tuple[str, int]]],
        List[List[Tuple[str, int]]],
    ]:
        batch, time_steps, _ = logits.shape
        if lengths.dim() != 1 or lengths.size(0) != batch:
            raise ValueError("lengths must have shape (batch,) for Viterbi decoding")

        sequences: List[List[int]] = [[] for _ in range(batch)]
        collapsed_tokens: List[List[int]] = [[] for _ in range(batch)]
        char_run_lengths: List[List[int]] = [[] for _ in range(batch)]
        frame_run_lengths: List[List[int]] = [[] for _ in range(batch)]
        collapsed_texts: List[str] = ["" for _ in range(batch)]
        rle_chars: List[List[Tuple[str, int]]] = [[] for _ in range(batch)]
        rle_frames: List[List[Tuple[str, int]]] = [[] for _ in range(batch)]

        valid_mask = lengths > 0
        if not valid_mask.any():
            return (
                sequences,
                collapsed_tokens,
                char_run_lengths,
                frame_run_lengths,
                collapsed_texts,
                rle_chars,
                rle_frames,
            )

        k2 = _require_k2()
        logits_f32 = logits.to(torch.float32)
        dense_inputs = build_blank_dense_inputs(logits_f32, lengths)

        valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
        dense_valid = dense_inputs.select(valid_indices)
        graph = build_blank_denominator_graph(transition)
        graph_vec = create_fsa_vec([graph.clone() for _ in range(valid_indices.numel())])
        lattice = k2.intersect_dense_pruned(
            graph_vec,
            dense_valid.to_dense_fsa(),
            search_beam=self.viterbi_search_beam,
            output_beam=self.viterbi_output_beam,
            min_active_states=self.viterbi_min_active_states,
            max_active_states=self.viterbi_max_active_states,
        )
        lattice.scores = -lattice.scores
        best_paths = k2.shortest_path(lattice, use_double_scores=False)

        valid_indices_cpu = valid_indices.detach().cpu().tolist()
        for local_idx, batch_idx in enumerate(valid_indices_cpu):
            idx_tensor = torch.tensor([local_idx], device=logits.device, dtype=torch.int32)
            single = k2.index_fsa(best_paths, idx_tensor)
            labels = single.labels.to(torch.long)
            label_values = [int(value) for value in labels.detach().cpu().tolist() if value >= 0]
            (
                run_bases,
                per_char_tokens,
                char_lengths,
                frame_lengths,
                collapsed_text,
                char_rle,
                frame_rle,
            ) = self._labels_to_blank_sequences(label_values)
            sequences[batch_idx] = run_bases
            collapsed_tokens[batch_idx] = per_char_tokens
            char_run_lengths[batch_idx] = char_lengths
            frame_run_lengths[batch_idx] = frame_lengths
            collapsed_texts[batch_idx] = collapsed_text
            rle_chars[batch_idx] = char_rle
            rle_frames[batch_idx] = frame_rle

        return (
            sequences,
            collapsed_tokens,
            char_run_lengths,
            frame_run_lengths,
            collapsed_texts,
            rle_chars,
            rle_frames,
        )

    def _labels_to_blank_sequences(
        self, labels: Sequence[int]
    ) -> Tuple[
        List[int],
        List[int],
        List[int],
        List[int],
        str,
        List[Tuple[str, int]],
        List[Tuple[str, int]],
    ]:
        raw_path = [int(label) for label in labels]
        run_bases: List[int] = []
        collapsed_tokens: List[int] = []
        char_run_lengths: List[int] = []
        collapsed_chars: List[str] = []
        frame_rle: List[Tuple[str, int]] = []

        last_output: Optional[int] = None
        frame_last_label: Optional[int] = None
        frame_length = 0

        for label in raw_path:
            if frame_last_label is None:
                frame_last_label = label
                frame_length = 1
            elif label == frame_last_label:
                frame_length += 1
            else:
                frame_rle.append((self._blank_label_to_char(frame_last_label), frame_length))
                frame_last_label = label
                frame_length = 1

            if label == BLANK_LABEL:
                last_output = None
                continue

            base_index = label - 1
            collapsed_tokens.append(base_index)
            if last_output == base_index:
                char_run_lengths[-1] += 1
            else:
                run_bases.append(base_index)
                char_run_lengths.append(1)
                collapsed_chars.append(BASE_VOCAB[base_index - 1])
                last_output = base_index

        if frame_last_label is not None and frame_length > 0:
            frame_rle.append((self._blank_label_to_char(frame_last_label), frame_length))

        collapsed_text = "".join(collapsed_chars)
        char_rle = list(zip(collapsed_chars, char_run_lengths))
        frame_run_lengths = [length for char, length in frame_rle if char != "-"]

        return (
            run_bases,
            collapsed_tokens,
            char_run_lengths,
            frame_run_lengths,
            collapsed_text,
            char_rle,
            frame_rle,
        )

    def _viterbi(
        self,
        emissions: Tensor,
        transition: Tensor,
        bos: Tensor,
        eos: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[
        List[List[int]],
        List[List[int]],
        List[List[int]],
        List[List[int]],
    ]:
        batch, time_steps, _ = emissions.shape

        if lengths is None:
            lengths_tensor = torch.full((batch,), time_steps, dtype=torch.long, device=emissions.device)
        else:
            if lengths.dim() != 1 or lengths.size(0) != batch:
                raise ValueError("lengths must be a 1D tensor matching the batch size")
            lengths_tensor = lengths.to(device=emissions.device, dtype=torch.long)

        sequences: List[List[int]] = [[] for _ in range(batch)]
        collapsed_tokens: List[List[int]] = [[] for _ in range(batch)]
        char_run_lengths: List[List[int]] = [[] for _ in range(batch)]
        frame_run_lengths: List[List[int]] = [[] for _ in range(batch)]

        valid_mask = lengths_tensor > 0
        if not valid_mask.any():
            return sequences, collapsed_tokens, char_run_lengths, frame_run_lengths

        k2 = _require_k2()
        emissions_f32 = emissions.to(torch.float32)
        transition_f32 = transition.to(dtype=torch.float32, device=emissions.device)
        bos_f32 = bos.to(dtype=torch.float32, device=emissions.device)
        eos_f32 = eos.to(dtype=torch.float32, device=emissions.device)

        dense_inputs = build_dense_inputs(emissions_f32, lengths_tensor)

        valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)
        dense_valid = dense_inputs.select(valid_indices)
        transition_graph = build_transition_graph(transition_f32, bos_f32, eos_f32)
        graph_vec = create_fsa_vec([transition_graph.clone() for _ in range(valid_indices.numel())])
        lattice = k2.intersect_dense_pruned(
            graph_vec,
            dense_valid.to_dense_fsa(),
            search_beam=self.viterbi_search_beam,
            output_beam=self.viterbi_output_beam,
            min_active_states=self.viterbi_min_active_states,
            max_active_states=self.viterbi_max_active_states,
        )
        lattice.scores = -lattice.scores
        best_paths = k2.shortest_path(lattice, use_double_scores=False)

        valid_indices_cpu = valid_indices.detach().cpu().tolist()
        for local_idx, batch_idx in enumerate(valid_indices_cpu):
            idx_tensor = torch.tensor([local_idx], device=emissions.device, dtype=torch.int32)
            single = k2.index_fsa(best_paths, idx_tensor)
            labels = single.labels.to(torch.long)
            label_values = labels.detach().cpu().tolist()
            state_sequence = [int(value) - 1 for value in label_values if value > 0]
            (
                decoded,
                per_char_tokens,
                char_lengths,
                frame_lengths,
            ) = self._states_to_sequence(state_sequence)
            sequences[batch_idx] = decoded
            collapsed_tokens[batch_idx] = per_char_tokens
            char_run_lengths[batch_idx] = char_lengths
            frame_run_lengths[batch_idx] = frame_lengths

        return sequences, collapsed_tokens, char_run_lengths, frame_run_lengths

    @staticmethod
    def _states_to_sequence(
        states: Sequence[int],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        decoded_runs: List[int] = []
        collapsed_tokens: List[int] = []
        char_run_lengths: List[int] = []
        frame_run_lengths: List[int] = []

        current_base: Optional[int] = None
        current_frame_length = 0
        current_char_length = 0

        def _flush_current() -> None:
            if current_base is None:
                return
            run_len_frames = max(current_frame_length, 1)
            run_len_chars = max(current_char_length, 1)
            decoded_runs.append(current_base)
            char_run_lengths.append(run_len_chars)
            frame_run_lengths.append(run_len_frames)

        for state in states:
            if state < STATE_EXTEND_OFFSET:
                base = state + 1
                collapsed_tokens.append(base)
                nonlocal_current = current_base
                if nonlocal_current is None:
                    current_base = base
                    current_frame_length = 1
                    current_char_length = 1
                    continue
                if base == nonlocal_current:
                    current_frame_length += 1
                    current_char_length += 1
                    continue
                _flush_current()
                current_base = base
                current_frame_length = 1
                current_char_length = 1
            else:
                if current_base is None:
                    # Skip stray extend without an enter; this should not happen but guard for robustness.
                    continue
                current_frame_length += 1

        if current_base is not None:
            _flush_current()

        return decoded_runs, collapsed_tokens, char_run_lengths, frame_run_lengths

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "embedding" not in batch:
            raise KeyError("Decoder expects 'embedding' in batch")

        hidden = batch["embedding"]
        if not torch.isfinite(hidden).all():
            hidden_logger = logging.getLogger("decoder")
            bad = ~torch.isfinite(hidden)
            bad_batches = bad.any(dim=(1, 2))
            indices = torch.nonzero(bad_batches, as_tuple=False).view(-1)
            for idx in indices.detach().cpu().tolist()[:8]:
                row = hidden[idx]
                hidden_logger.warning(
                    "decoder input hidden non-finite | b=%d | any_finite=%s | min=%.3e max=%.3e",
                    idx,
                    torch.isfinite(row).any().item(),
                    torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).min().item(),
                    torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).max().item(),
                )
        if hidden.dim() != 3:
            raise ValueError("Decoder expects embedding tensor of shape (B, T, D)")
        if hidden.size(-1) != self.model_dim:
            raise ValueError(
                f"Decoder configured for model_dim={self.model_dim}, received {hidden.size(-1)}"
            )

        key_padding_mask = None
        lengths = None
        pad_mask = batch.get("decoder_padding_mask")
        if pad_mask is not None:
            key_padding_mask = pad_mask.to(device=hidden.device, dtype=torch.bool)
            lengths = (~key_padding_mask).sum(dim=1).to(dtype=torch.long)

        tap_encoded_grad = os.environ.get("PF2_TAP_DECODER_ENC_GRAD", "0") == "1"
        encoded = self.transformer(hidden, key_padding_mask=key_padding_mask)
        if tap_encoded_grad and encoded.requires_grad:
            grad_logger = logging.getLogger("decoder")

            def _encoded_hook(grad: Tensor) -> Tensor:
                grad_safe = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                if grad_safe.numel() == 0:
                    g_min = 0.0
                    g_max = 0.0
                    g_norm = 0.0
                else:
                    g_min = grad_safe.min().item()
                    g_max = grad_safe.max().item()
                    g_norm = torch.linalg.vector_norm(grad_safe).item()
                grad_logger.warning(
                    "grad[encoded] | any_finite=%s | any_nan=%s any_inf=%s | min=%.3e max=%.3e | ||g||=%.3e",
                    bool(torch.isfinite(grad).any().item()),
                    bool(torch.isnan(grad).any().item()),
                    bool(torch.isinf(grad).any().item()),
                    float(g_min),
                    float(g_max),
                    float(g_norm),
                )
                return grad

            encoded.register_hook(_encoded_hook)
        # Guard against non-finite activations leaking into the emission heads.
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1e4, neginf=-1e4)
        batch_size, time_steps, _ = encoded.shape
        if lengths is None:
            lengths_tensor = torch.full(
                (batch_size,), time_steps, device=encoded.device, dtype=torch.long
            )
        else:
            lengths_tensor = lengths.to(device=encoded.device, dtype=torch.long)

        if self.variant == "ctc_crf_blank":
            return self._forward_blank_variant(
                batch, encoded, key_padding_mask, lengths_tensor
            )

        emissions = self._compute_emissions(encoded)
        emissions = torch.nan_to_num(emissions, nan=0.0, posinf=1e4, neginf=-1e4)
        transition = self._build_transition_matrix(emissions.dtype, emissions.device)

        bos = self.bos_scores.clamp(min=-self.clamp_transition, max=self.clamp_transition)
        eos = self.eos_scores.clamp(min=-self.clamp_transition, max=self.clamp_transition)
        # Ensure BOS/EOS scores are finite at runtime (optimizers can corrupt params if grads are non-finite).
        if not torch.isfinite(bos).all() or not torch.isfinite(eos).all():
            bos = torch.nan_to_num(bos, nan=0.0, posinf=1e4, neginf=-1e4)
            eos = torch.nan_to_num(eos, nan=0.0, posinf=1e4, neginf=-1e4)

        output = dict(batch)
        output["embedding"] = encoded
        output["ctc_emissions"] = emissions
        output["ctc_transition"] = transition
        output["ctc_bos"] = bos
        output["ctc_eos"] = eos
        output["ctc_variant"] = "ctc_crf_enter_extend"
        output["decoder_lengths"] = lengths_tensor

        if self.return_viterbi:
            (
                sequences,
                collapsed_tokens,
                char_run_lengths,
                frame_run_lengths,
            ) = self._viterbi(emissions, transition, bos, eos, lengths_tensor)
            output["viterbi_sequence"] = sequences
            output["viterbi_collapsed_tokens"] = collapsed_tokens
            output["viterbi_char_run_length"] = char_run_lengths
            output["viterbi_frame_run_length"] = frame_run_lengths
            # Backward compatibility with legacy callers expecting 'viterbi_run_length'.
            output["viterbi_run_length"] = frame_run_lengths

        if key_padding_mask is not None:
            output.setdefault("decoder_padding_mask", key_padding_mask)

        return output


@DECODER_REGISTRY.register("ctc_crf")
def build_ctc_crf_decoder(**kwargs: object) -> DecoderBase:
    return CTCCRFDecoder(**kwargs)


__all__ = ["CTCCRFDecoder", "build_ctc_crf_decoder"]