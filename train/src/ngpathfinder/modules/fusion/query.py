"""DETR-style query fusion with cross-attention and auxiliary signals."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import FUSION_REGISTRY, FusionBase
from ngpathfinder.losses.ctc_fast import minimal_ctc_input_length


def _build_fourier_features(
    positions: Tensor,
    max_frequency: int,
) -> Tensor:
    """Return sinusoidal features for the given normalized positions.

    Args:
        positions: Normalized query positions in ``[0, 1]`` with shape ``(Q,)``.
        max_frequency: Maximum frequency ``f`` used to create ``sin(2π f τ)`` and
            ``cos(2π f τ)`` features.

    Returns:
        Tensor containing concatenated ``[τ, τ², sin, cos]`` features of shape
        ``(Q, feature_dim)``.
    """

    if positions.dim() != 1:
        raise ValueError("positions must be 1-dimensional")

    features = [positions.unsqueeze(-1), positions.pow(2).unsqueeze(-1)]
    if max_frequency > 0:
        frequencies = torch.arange(
            1, max_frequency + 1, device=positions.device, dtype=positions.dtype
        )
        angles = 2.0 * torch.pi * positions.unsqueeze(-1) * frequencies
        features.append(torch.sin(angles))
        features.append(torch.cos(angles))
    return torch.cat(features, dim=-1)

class _QuerySelfAttentionBlock(nn.Module):
    """Pre-LN Transformer block for query self-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        ffn_dim = int(hidden_dim * ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        attn_input = self.norm1(x)
        residual = x
        attn_output = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = residual + self.dropout(attn_output)

        residual = x
        ffn_input = self.norm2(x)
        x = residual + self.ffn(ffn_input)
        return x


class QueryFusion(FusionBase):
    """Fuse multi-read encodings using learnable queries and cross-attention."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 320,
        max_learned_queries: Optional[int] = None,
        dynamic_query_cap: Optional[int] = None,
        funq_hidden_dim: Optional[int] = None,
        fourier_frequencies: int = 32,
        use_learned_residual: bool = True,
        hint_strength: float = 0.2,
        gate_threshold: float = 0.5,
        use_gating: bool = True,
        relative_bias: str = "gaussian",
        gaussian_sigma_init: Tuple[float, ...] = (0.04, 0.08, 0.16),
        window_multiplier: float = 2.5,
        min_window_bins: int = 3,
        summary_dim: Optional[int] = None,
        affine_min_scale: float = 1e-3,
        eps: float = 1e-6,
        store_attention: bool = False,
        use_kv_positional_encoding: bool = False,
        kv_position_scale: float = 0.1,
        num_query_self_layers: int = 0,
        query_self_dropout: float = 0.1,
        query_self_ffn_multiplier: float = 4.0,
        use_identity_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        if gaussian_sigma_init and len(gaussian_sigma_init) == 0:
            raise ValueError("gaussian_sigma_init must not be empty")
        if num_query_self_layers < 0:
            raise ValueError("num_query_self_layers must be >= 0")
        if query_self_dropout < 0.0 or query_self_dropout > 1.0:
            raise ValueError("query_self_dropout must be in [0, 1]")
        if query_self_ffn_multiplier <= 0:
            raise ValueError("query_self_ffn_multiplier must be positive")
        if window_multiplier <= 0.0:
            raise ValueError("window_multiplier must be positive")
        if min_window_bins < 1:
            raise ValueError("min_window_bins must be >= 1")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_query_self_layers = num_query_self_layers
        self.query_self_dropout = query_self_dropout
        self.query_self_ffn_multiplier = query_self_ffn_multiplier
        if max_learned_queries is None:
            self.max_learned_queries = num_queries
        else:
            if max_learned_queries < num_queries:
                raise ValueError("max_learned_queries must be >= num_queries")
            self.max_learned_queries = max_learned_queries
        if dynamic_query_cap is not None and dynamic_query_cap < 1:
            raise ValueError("dynamic_query_cap must be positive when provided")
        self.dynamic_query_cap = dynamic_query_cap
        self.fourier_frequencies = fourier_frequencies
        self.use_learned_residual = use_learned_residual
        self.hint_strength = nn.Parameter(torch.tensor(hint_strength, dtype=torch.float32))
        self.use_gating = use_gating
        self.gate_threshold = gate_threshold
        self.relative_bias = relative_bias
        self.affine_min_scale = affine_min_scale
        self.eps = eps
        self.store_attention = store_attention
        self.use_kv_positional_encoding = use_kv_positional_encoding
        self.use_identity_cross_attention = use_identity_cross_attention
        self.window_multiplier = float(window_multiplier)
        self.min_window_bins = int(min_window_bins)

        feature_dim = 2 + 2 * max(fourier_frequencies, 0)
        funq_hidden_dim = funq_hidden_dim or hidden_dim

        self.funq_mlp = nn.Sequential(
            nn.Linear(feature_dim, funq_hidden_dim),
            nn.SiLU(),
            nn.Linear(funq_hidden_dim, hidden_dim),
        )
        self.funq_norm = nn.LayerNorm(hidden_dim)
        self.position_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.position_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        if use_kv_positional_encoding:
            self.kv_position_scale = nn.Parameter(
                torch.tensor(kv_position_scale, dtype=torch.float32)
            )
        else:
            self.register_parameter("kv_position_scale", None)

        if use_learned_residual:
            self.learned_query = nn.Parameter(torch.zeros(self.max_learned_queries, hidden_dim))
            self.learned_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.register_parameter("learned_query", None)
            self.register_parameter("learned_scale", None)

        if self.use_gating:
            gate_hidden = max(hidden_dim // 4, 16)
            self.gate_mlp = nn.Sequential(
                nn.Linear(feature_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )
        else:
            self.gate_mlp = None

        head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.scale = head_dim ** -0.5

        if num_query_self_layers > 0:
            self.query_self_layers = nn.ModuleList(
                [
                    _QuerySelfAttentionBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=query_self_dropout,
                        ffn_multiplier=query_self_ffn_multiplier,
                    )
                    for _ in range(num_query_self_layers)
                ]
            )
        else:
            self.query_self_layers = nn.ModuleList()

        if relative_bias not in {"gaussian", "alibi", "none", "hard_window"}:
            raise ValueError(
                "relative_bias must be one of {'gaussian', 'alibi', 'none', 'hard_window'}"
            )

        if relative_bias in {"gaussian", "hard_window"}:
            sigma = torch.tensor(gaussian_sigma_init, dtype=torch.float32)
            if sigma.numel() < num_heads:
                repeats = (num_heads + sigma.numel() - 1) // sigma.numel()
                sigma = sigma.repeat(repeats)[:num_heads]
            self.gaussian_log_sigma = nn.Parameter(torch.log(sigma))
        else:
            self.register_parameter("gaussian_log_sigma", None)

        if relative_bias == "alibi":
            slopes = torch.arange(1, num_heads + 1, dtype=torch.float32)
            slopes = slopes / slopes.max()
            self.alibi_slopes = nn.Parameter(slopes)
        else:
            self.register_parameter("alibi_slopes", None)

        if summary_dim is not None:
            self.summary_dim = summary_dim
            self.affine_proj = nn.Sequential(
                nn.Linear(summary_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )
            self.read_weight_proj = nn.Sequential(
                nn.Linear(summary_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.summary_dim = None
            self.affine_proj = None
            self.read_weight_proj = None

    def _prepare_queries(self, query_count: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
        positions = torch.linspace(0.0, 1.0, query_count, device=device, dtype=dtype)
        features = _build_fourier_features(positions, self.fourier_frequencies)
        base = self.funq_norm(
            self.funq_mlp(features) + self.position_scale * self.position_proj(features)
        )

        if self.use_learned_residual and self.learned_query is not None:
            learned = self.learned_query.transpose(0, 1).unsqueeze(0)  # (1, hidden, Qmax)
            if query_count != self.max_learned_queries:
                learned = F.interpolate(
                    learned,
                    size=query_count,
                    mode="linear",
                    align_corners=True,
                )
            learned = learned.squeeze(0).transpose(0, 1)
            queries = base + self.learned_scale * learned
        else:
            queries = base

        gate = None
        if self.use_gating and self.gate_mlp is not None:
            gate_logits = self.gate_mlp(features)
            gate = torch.sigmoid(gate_logits)

        return queries, gate, positions

    def _reshape_heads(self, tensor: Tensor, seq_len: int) -> Tensor:
        batch = tensor.size(0)
        return (
            tensor.view(batch, seq_len, self.num_heads, self.hidden_dim // self.num_heads)
            .transpose(1, 2)
            .contiguous()
        )

    def _build_kv_positional_features(self, normalized_positions: Tensor) -> Tensor:
        """Create Fourier positional features for keys/values.

        Args:
            normalized_positions: Tensor with shape ``(B×R, T)`` containing
                normalized time coordinates in ``[0, 1]``.

        Returns:
            Tensor of shape ``(B×R, T, feature_dim)`` with the same Fourier feature
            construction used for query preparation.
        """

        features = [normalized_positions.unsqueeze(-1), normalized_positions.pow(2).unsqueeze(-1)]
        if self.fourier_frequencies > 0:
            frequencies = torch.arange(
                1,
                self.fourier_frequencies + 1,
                device=normalized_positions.device,
                dtype=normalized_positions.dtype,
            )
            angles = 2.0 * torch.pi * normalized_positions.unsqueeze(-1) * frequencies.view(1, 1, -1)
            features.append(torch.sin(angles))
            features.append(torch.cos(angles))
        return torch.cat(features, dim=-1)

    def _compute_relative_bias(
        self,
        normalized_q: Tensor,
        normalized_t: Tensor,
        lengths: Tensor,
        summary: Optional[Tensor],
    ) -> Optional[Tensor]:
        if self.relative_bias == "none":
            return None

        batch = normalized_t.size(0)
        normalized_q = normalized_q.view(1, -1, 1).expand(batch, -1, -1)
        normalized_t = normalized_t.view(batch, 1, -1)

        if self.affine_proj is not None and summary is not None:
            affine = self.affine_proj(summary)
            slope = F.softplus(affine[..., 0]) + self.affine_min_scale
            intercept = affine[..., 1]
            slope = slope.unsqueeze(-1).unsqueeze(-1)
            intercept = intercept.unsqueeze(-1).unsqueeze(-1)
            centers = slope * normalized_q + intercept
            centers = centers.clamp(0.0, 1.0)
        else:
            centers = normalized_q

        if self.relative_bias == "hard_window":
            if self.gaussian_log_sigma is None:
                raise RuntimeError("gaussian_log_sigma is not initialized for hard_window bias")
            sigma = torch.exp(self.gaussian_log_sigma).view(1, self.num_heads, 1, 1)
            radius = self.window_multiplier * sigma

            if self.min_window_bins > 0:
                valid_len = lengths.view(batch, 1, 1, 1)
                effective_span = torch.clamp(valid_len - 1.0, min=1.0)
                min_radius_norm = (self.min_window_bins / effective_span).to(radius.dtype)
                radius = torch.maximum(radius, min_radius_norm)

            t_positions = normalized_t.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            center_positions = centers.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            diff = (t_positions - center_positions).abs()
            bias = diff.new_full(diff.shape, float("-inf"))
            bias = bias.masked_fill(diff <= radius, 0.0)
            return bias

        if self.relative_bias == "gaussian":
            sigma = torch.exp(self.gaussian_log_sigma)
            sigma = sigma.view(1, self.num_heads, 1, 1)
            diff = normalized_t.unsqueeze(1) - centers.unsqueeze(1)
            bias = -(diff.pow(2)) / (2 * sigma.pow(2).clamp_min(self.eps))
            return bias

        if self.relative_bias == "alibi":
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            diff = torch.abs(normalized_t.unsqueeze(1) - centers.unsqueeze(1))
            bias = -slopes * diff
            return bias

        return None

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = dict(batch)
        embedding = batch["embedding"]
        hard_mask = batch.get("hard_mask")
        soft_hint = batch.get("soft_hint")
        mask_provided = hard_mask is not None

        if embedding.dim() == 3:
            embedding = embedding.unsqueeze(1)
            if hard_mask is not None:
                hard_mask = hard_mask.unsqueeze(1)
            if soft_hint is not None:
                soft_hint = soft_hint.unsqueeze(1)
        elif embedding.dim() != 4:
            raise ValueError("embedding must have shape (B, T, D) or (B, R, T, D)")

        batch_size, num_reads, time_steps, hidden_dim = embedding.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected embedding hidden dim {self.hidden_dim}, but received {hidden_dim}"
            )

        if hard_mask is None:
            hard_mask = torch.ones(
                batch_size, num_reads, time_steps, device=embedding.device, dtype=embedding.dtype
            )
        else:
            hard_mask = hard_mask.to(device=embedding.device, dtype=embedding.dtype)
        if soft_hint is None:
            soft_hint = hard_mask.to(device=embedding.device, dtype=embedding.dtype)

        if batch.get("encoder_embedding") is None:
            batch["encoder_embedding"] = embedding.detach()
            batch["encoder_hard_mask"] = hard_mask.detach()
            batch["encoder_soft_hint"] = soft_hint.detach()

        read_mask = (hard_mask.sum(dim=-1) > 0).to(dtype=embedding.dtype)

        requested_count = batch.get("target_query_count", self.num_queries)
        if torch.is_tensor(requested_count):
            if requested_count.numel() != 1:
                raise ValueError("target_query_count tensor must contain a single value")
            requested_count = int(requested_count.item())
        else:
            requested_count = int(requested_count)

        if requested_count <= 0:
            requested_count = 1

        base_query_count = max(requested_count, self.num_queries)
        cap = self.dynamic_query_cap if self.dynamic_query_cap is not None else self.max_learned_queries
        cap = max(cap, self.num_queries)
        query_count = min(base_query_count, self.max_learned_queries, cap)
        queries, gate, normalized_q = self._prepare_queries(
            query_count, device=embedding.device, dtype=embedding.dtype
        )

        flat_embedding = embedding.view(batch_size * num_reads, time_steps, hidden_dim)
        flat_mask = hard_mask.view(batch_size * num_reads, time_steps)
        flat_hint = soft_hint.view(batch_size * num_reads, time_steps)

        queries = queries.to(embedding.device, embedding.dtype)
        if self.use_identity_cross_attention:
            residual_queries = queries
            normed_queries = self.norm_q(queries)
        else:
            residual_queries = None
            normed_queries = self.norm_q(queries)
        expanded_queries = normed_queries.unsqueeze(0).expand(flat_embedding.size(0), -1, -1)

        keys = self.norm_k(flat_embedding)
        values = self.norm_v(flat_embedding)

        mask_bool = flat_mask <= 0

        # Ensure length normalization handles boolean or integer masks by using a
        # floating representation before applying clamp operations.
        valid_lengths = flat_mask.sum(dim=-1).to(embedding.dtype)
        time_positions = torch.arange(time_steps, device=embedding.device, dtype=embedding.dtype)
        denom = (valid_lengths - 1).clamp_min(1.0).unsqueeze(-1)
        normalized_t = time_positions.unsqueeze(0) / denom
        if self.use_kv_positional_encoding:
            kv_positions = normalized_t.clamp(0.0, 1.0)
            kv_features = self._build_kv_positional_features(kv_positions)
            kv_encoding = self.position_proj(kv_features)
            if self.kv_position_scale is not None:
                kv_encoding = kv_encoding * self.kv_position_scale.to(dtype=kv_encoding.dtype)
            keys = keys + kv_encoding
            values = values + kv_encoding

        q_proj = self._reshape_heads(self.q_proj(expanded_queries), query_count)
        k_proj = self._reshape_heads(self.k_proj(keys), time_steps)
        v_proj = self._reshape_heads(self.v_proj(values), time_steps)

        summary = None
        if self.summary_dim is not None and "read_summary" in batch:
            summary = batch["read_summary"].view(batch_size * num_reads, -1)
            if summary.size(-1) != self.summary_dim:
                raise ValueError(
                    f"read_summary last dimension {summary.size(-1)} does not match configured summary_dim {self.summary_dim}"
                )

        attn_logits = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale
        bias = self._compute_relative_bias(normalized_q, normalized_t, valid_lengths, summary)
        if bias is not None:
            attn_logits = attn_logits + bias

        if flat_hint is not None:
            attn_logits = attn_logits + self.hint_strength * flat_hint.view(
                flat_hint.size(0), 1, 1, flat_hint.size(-1)
            )

        if gate is not None:
            gate = gate.to(embedding.device, embedding.dtype)
            attn_logits = attn_logits + torch.log(gate.clamp_min(self.eps)).view(1, 1, -1, 1)

        if mask_bool.any():
            attn_logits = attn_logits.masked_fill(
                mask_bool.view(mask_bool.size(0), 1, 1, time_steps), float("-inf")
            )

        invalid_rows = mask_bool.all(dim=-1)
        if invalid_rows.any():
            safe_logits = attn_logits.clone()
            safe_logits[invalid_rows] = 0.0
            attn_probs = torch.softmax(safe_logits, dim=-1)
            valid_scaling = (~invalid_rows).to(attn_probs.dtype).view(-1, 1, 1, 1)
            attn_probs = attn_probs * valid_scaling
        else:
            attn_probs = torch.softmax(attn_logits, dim=-1)

        cached_attention: Optional[Tensor] = None
        if self.store_attention:
            cached_attention = attn_probs.detach().to(torch.float32)

        # Determine if we can leverage the fused SDPA kernel. Only safe when no
        # extra bias/gating/hints are applied so the numerical behaviour matches
        # the original implementation.
        hint_strength_value = float(self.hint_strength.detach().to(torch.float32).item())
        use_sdpa = (
            self.relative_bias == "none"
            and not getattr(self, "use_gating", False)
            and (flat_hint is None or hint_strength_value == 0.0)
            and not self.store_attention
            and os.getenv("PF2_DISABLE_SDPA", "0") != "1"
        )

        if use_sdpa:
            mask = mask_bool.to(device=embedding.device)
            bxr, tgt_steps = mask.shape
            head_dim = hidden_dim // self.num_heads

            q = q_proj.view(bxr, query_count, self.num_heads, head_dim).transpose(1, 2).reshape(
                bxr * self.num_heads, query_count, head_dim
            )
            k = k_proj.view(bxr, time_steps, self.num_heads, head_dim).transpose(1, 2).reshape(
                bxr * self.num_heads, time_steps, head_dim
            )
            v = v_proj.view(bxr, time_steps, self.num_heads, head_dim).transpose(1, 2).reshape(
                bxr * self.num_heads, time_steps, head_dim
            )

            attn_mask_bool = mask.unsqueeze(1).expand(bxr, query_count, tgt_steps)
            attn_mask_bool = attn_mask_bool.repeat_interleave(self.num_heads, dim=0)
            attn_mask = torch.zeros(
                attn_mask_bool.shape,
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask = attn_mask.masked_fill(attn_mask_bool, float("-inf"))

            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

            context = (
                context.view(bxr, self.num_heads, query_count, head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bxr, query_count, hidden_dim)
            )

            if not hasattr(self, "_sdpa_notified"):
                message = "QueryFusion: using SDPA fast path"
                logging.getLogger("fusion.query").info(message)
                print(message, flush=True)
                self._sdpa_notified = True
        else:
            attn_weights = self.dropout(attn_probs)
            context = torch.matmul(attn_weights, v_proj)

        invalid_rows = mask_bool.all(dim=-1)
        if invalid_rows.any():
            context = context.clone()
            context[invalid_rows] = 0.0
        context = context.transpose(1, 2).contiguous().view(flat_embedding.size(0), query_count, hidden_dim)
        residual_context = self.out_proj(context)

        if self.query_self_layers:
            if gate is not None:
                gate_mask = gate.view(1, query_count).detach() <= self.gate_threshold
                gate_mask = gate_mask.expand(context.size(0), -1)
            else:
                gate_mask = None

            if requested_count < query_count:
                dynamic_mask = (
                    torch.arange(query_count, device=context.device).view(1, -1)
                    >= requested_count
                )
                dynamic_mask = dynamic_mask.expand(context.size(0), -1)
                if gate_mask is None:
                    gate_mask = dynamic_mask
                else:
                    gate_mask = gate_mask | dynamic_mask

            if gate_mask is not None:
                invalid_rows = gate_mask.all(dim=1)
            else:
                invalid_rows = None

            if gate_mask is not None:
                all_invalid = (
                    bool(invalid_rows.all().item())
                    if invalid_rows is not None and invalid_rows.numel() > 0
                    else False
                )
                if invalid_rows is not None and invalid_rows.any() and not all_invalid:
                    valid_rows = ~invalid_rows
                    residual_context = residual_context.clone()
                    valid_context = residual_context[valid_rows]
                    valid_mask = gate_mask[valid_rows]
                    for block in self.query_self_layers:
                        valid_context = block(valid_context, key_padding_mask=valid_mask)
                    residual_context[valid_rows] = valid_context
                elif not all_invalid:
                    for block in self.query_self_layers:
                        residual_context = block(residual_context, key_padding_mask=gate_mask)
            else:
                for block in self.query_self_layers:
                    residual_context = block(residual_context, key_padding_mask=None)

        if gate is not None:
            residual_context = residual_context * gate.view(1, query_count, 1)

        if self.use_identity_cross_attention and residual_queries is not None:
            expanded_residual = residual_queries.unsqueeze(0).expand(residual_context.size(0), -1, -1)
            if gate is not None:
                expanded_gate = gate.view(1, query_count, 1)
                identity_context = expanded_residual * expanded_gate
            else:
                identity_context = expanded_residual
            context = identity_context + residual_context
        else:
            context = residual_context

        read_outputs = context.view(batch_size, num_reads, query_count, hidden_dim)

        if self.read_weight_proj is not None and "read_summary" in batch:
            read_summary = batch["read_summary"]
            if read_summary.size(-1) != self.summary_dim:
                raise ValueError(
                    f"read_summary last dimension {read_summary.size(-1)} does not match configured summary_dim {self.summary_dim}"
                )
            read_scores = self.read_weight_proj(read_summary).squeeze(-1)
            read_weight_values = torch.softmax(read_scores, dim=1)
        else:
            read_mask_expanded = read_mask.to(device=embedding.device, dtype=embedding.dtype)
            valid_counts = read_mask_expanded.sum(dim=1, keepdim=True)
            safe_counts = valid_counts.clamp_min(self.eps)
            read_weight_values = read_mask_expanded / safe_counts

            if (valid_counts <= 0).any():
                uniform = torch.full(
                    (batch_size, num_reads),
                    1.0 / num_reads,
                    device=embedding.device,
                    dtype=embedding.dtype,
                )
                read_weight_values = torch.where(
                    (valid_counts > 0),
                    read_weight_values,
                    uniform,
                )

        read_weights = read_weight_values.view(batch_size, num_reads, 1, 1)

        fused = (read_outputs * read_weights).sum(dim=1)

        if gate is not None:
            gate_values = gate.view(1, query_count).expand(batch_size, -1)
        else:
            gate_values = torch.ones(
                batch_size,
                query_count,
                device=embedding.device,
                dtype=embedding.dtype,
            )

        gate_values = gate_values.clone()

        hard_mask_out = (gate_values.detach() > self.gate_threshold).to(embedding.dtype)

        min_queries: Tensor
        if mask_provided:
            valid_steps = (hard_mask > 0).sum(dim=-1)
            max_valid_steps = valid_steps.max(dim=1).values
            total_steps = float(hard_mask.size(-1)) if hard_mask.size(-1) > 0 else 1.0
            min_queries = torch.ceil((max_valid_steps / total_steps) * query_count).clamp(
                min=1, max=query_count
            ).to(dtype=torch.long)
        else:
            min_queries = torch.ones(batch_size, device=embedding.device, dtype=torch.long)

        reference_index = batch.get("reference_index")
        if isinstance(reference_index, torch.Tensor):
            ref = reference_index
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            if ref.dim() == 2 and ref.size(0) == batch_size:
                lengths_tensor = batch.get("reference_lengths")
                lengths: Optional[Tensor]
                if isinstance(lengths_tensor, torch.Tensor):
                    lengths = lengths_tensor.to(device=embedding.device, dtype=torch.long)
                    if lengths.dim() == 0:
                        lengths = lengths.unsqueeze(0)
                    else:
                        lengths = lengths.view(-1)
                else:
                    lengths = None

                if lengths is None or lengths.numel() < batch_size:
                    lengths = torch.full(
                        (batch_size,),
                        ref.size(1),
                        device=embedding.device,
                        dtype=torch.long,
                    )
                else:
                    lengths = lengths[:batch_size]

                minimal_budget = []
                for sample_idx in range(batch_size):
                    valid_len = int(lengths[sample_idx].item())
                    minimal_budget.append(minimal_ctc_input_length(ref[sample_idx], valid_len))

                minimal_tensor = torch.tensor(
                    minimal_budget, device=embedding.device, dtype=torch.long
                )
                minimal_tensor = minimal_tensor.clamp_min(1).clamp_max(query_count)
                min_queries = torch.maximum(min_queries, minimal_tensor)

        valid_queries = hard_mask_out.sum(dim=1, keepdim=True)
        fallback = valid_queries <= 0
        if fallback.any():
            hard_mask_out = torch.where(
                fallback.expand_as(hard_mask_out),
                torch.ones_like(hard_mask_out),
                hard_mask_out,
            )
            gate_values = torch.where(
                fallback.expand_as(gate_values),
                torch.ones_like(gate_values),
                gate_values,
            )
            valid_queries = hard_mask_out.sum(dim=1, keepdim=True)

        needs_min = valid_queries.squeeze(1).long() < min_queries
        if needs_min.any():
            threshold = torch.tensor(1.0, device=embedding.device, dtype=embedding.dtype)
            indices = torch.nonzero(needs_min, as_tuple=False).view(-1)
            for idx in indices.tolist():
                required = int(min_queries[idx].item())
                if required >= query_count:
                    hard_mask_out[idx] = torch.ones_like(hard_mask_out[idx])
                    gate_values[idx] = torch.ones_like(gate_values[idx])
                    continue

                required = max(required, 1)
                scores = gate_values[idx]
                topk = torch.topk(scores, k=required)
                mask_row = torch.zeros_like(scores)
                mask_row[topk.indices] = 1.0
                gate_row = scores.clone()
                gate_row[topk.indices] = threshold
                hard_mask_out[idx] = mask_row
                gate_values[idx] = gate_row

        batch["decoder_padding_mask"] = (hard_mask_out <= 0).to(torch.bool)
        soft_mask_out = gate_values

        attn_mean = attn_probs.mean(dim=1)
        attn_mean = attn_mean.view(batch_size, num_reads, query_count, time_steps)
        normalized_t_view = normalized_t.view(batch_size, num_reads, 1, time_steps)
        soft_position = (attn_mean * normalized_t_view).sum(dim=-1)
        coverage = attn_mean.sum(dim=2)

        duration_target: Optional[Tensor] = None
        move_tensor = batch.get("move")
        if isinstance(move_tensor, Tensor):
            move_expected_shape = (batch_size, num_reads, time_steps)
            if move_tensor.shape == move_expected_shape:
                move_float = move_tensor.to(device=embedding.device, dtype=attn_mean.dtype)
                per_read_duration = (attn_mean.detach() * move_float.unsqueeze(2)).sum(dim=-1)
                read_weight_values_detached = read_weight_values.detach().to(
                    device=embedding.device, dtype=attn_mean.dtype
                )
                duration_target = (
                    per_read_duration * read_weight_values_detached.unsqueeze(-1)
                ).sum(dim=1)
                duration_target = duration_target * hard_mask_out.detach()
                duration_target = duration_target.clamp_(min=0.0, max=1.0)
            else:
                logging.getLogger("fusion").warning(
                    "move tensor shape mismatch for duration target computation: expected (B=%d, R=%d, T=%d), got %s",
                    batch_size,
                    num_reads,
                    time_steps,
                    tuple(move_tensor.shape),
                )

        batch["fusion_embedding_per_read"] = read_outputs
        batch["fusion_attention"] = attn_mean
        if cached_attention is not None:
            batch["fusion_attention_full"] = cached_attention.view(
                batch_size,
                num_reads,
                self.num_heads,
                query_count,
                time_steps,
            )
        batch["fusion_soft_position"] = soft_position
        batch["fusion_coverage"] = coverage
        batch["fusion_gate"] = gate_values
        batch["fusion_read_weights"] = read_weight_values
        batch["fusion_read_mask"] = read_mask

        batch["embedding"] = fused
        batch["hard_mask"] = hard_mask_out
        batch["soft_hint"] = soft_mask_out
        if duration_target is not None:
            batch["duration_target"] = duration_target

        if not torch.isfinite(batch["embedding"]).all():
            fusion_logger = logging.getLogger("fusion")
            fusion_logger.warning("fusion embedding non-finite")

        return batch

    def set_store_attention(self, enabled: bool) -> None:
        """Toggle storing per-head attention matrices during ``forward``.

        Args:
            enabled: If ``True`` the module will attach
                ``batch["fusion_attention_full"]`` (shape ``B×R×H×Q×T``) with the
                detached attention matrices for debugging/visualisation. Setting it
                to ``False`` disables the extra allocation.
        """

        self.store_attention = bool(enabled)

    @contextmanager
    def capture_attention(self, enabled: bool = True):
        """Context manager to temporarily enable attention capture.

        This allows users to inspect attention weights in targeted portions of
        training without permanently enabling the feature.
        """

        previous = self.store_attention
        self.store_attention = bool(enabled)
        try:
            yield
        finally:
            self.store_attention = previous


@FUSION_REGISTRY.register("query")
def build_query_fusion(**kwargs: Any) -> FusionBase:
    return QueryFusion(**kwargs)


__all__ = ["QueryFusion", "build_query_fusion"]