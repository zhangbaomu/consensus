"""Set Transformer style aggregator with strand flag injection."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .base import AGGREGATOR_REGISTRY, AggregatorBase


def _expand_batch_time(tensor: Tensor, query_count: int) -> Tensor:
    """Repeat a ``(B, R, ...)`` tensor across the query dimension."""

    if tensor.dim() < 2:
        raise ValueError("Expected at least a (B, R) tensor for expansion")
    batch = tensor.size(0)
    expanded = tensor.unsqueeze(1).expand(batch, query_count, *tensor.shape[1:])
    return expanded.reshape(batch * query_count, *tensor.shape[1:])


def _safe_softmax(logits: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Compute a mask-aware softmax that avoids NaNs for fully-masked rows."""

    probs = torch.softmax(logits, dim=-1)
    if mask is None:
        return probs
    probs = probs.masked_fill(mask, 0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return probs / denom


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting additive biases and key masking."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
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
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply attention and optionally return the per-head weights."""

        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if bias is not None:
            attn = attn + bias

        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask, float("-inf"))

        attn_probs = _safe_softmax(attn, mask)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        output = self.out_proj(context)

        if need_weights:
            return output, attn_probs
        return output, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SetAttentionBlock(nn.Module):
    """Pre-LN self-attention block with optional bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, int(hidden_dim * ffn_multiplier), dropout)

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Optional[Tensor],
        bias: Optional[Tensor],
    ) -> Tensor:
        residual = x
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            bias=bias,
            need_weights=False,
        )
        x = residual + attn_output

        residual = x
        ff_input = self.norm2(x)
        x = residual + self.ff(ff_input)
        return x


class PoolingByMultiheadAttention(nn.Module):
    """Pooling with learnable seed queries."""

    def __init__(self, hidden_dim: int, num_heads: int, num_seeds: int, dropout: float) -> None:
        super().__init__()
        self.num_seeds = num_seeds
        self.seed = nn.Parameter(torch.randn(num_seeds, hidden_dim))
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Optional[Tensor],
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch = x.size(0)
        seed = self.seed.unsqueeze(0).expand(batch, -1, -1)
        output, weights = self.attn(
            seed,
            x,
            x,
            key_padding_mask=key_padding_mask,
            bias=None,
            need_weights=need_weights,
        )
        return output, weights


class SetTransformerAggregator(AggregatorBase):
    """Aggregate per-read embeddings with Set Transformer blocks and PMA."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        num_seeds: int = 1,
        flag_dim: int = 8,
        include_uncertainty: bool = True,
        residual_scale_init: float = 0.4,
        bias_strength: float = 1.0,
        gaussian_sigma_init: Tuple[float, ...] = (0.06, 0.12, 0.24),
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_seeds < 1:
            raise ValueError("num_seeds must be >= 1")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.include_uncertainty = include_uncertainty
        self.bias_strength = bias_strength

        self.blocks = nn.ModuleList(
            [
                SetAttentionBlock(hidden_dim, num_heads, ffn_multiplier, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pma = PoolingByMultiheadAttention(hidden_dim, num_heads, num_seeds, dropout)
        self.k_proj = nn.Linear(hidden_dim * num_seeds, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.flag_embed = nn.Embedding(2, flag_dim)
        self.flag_proj = nn.Sequential(
            nn.Linear(hidden_dim + flag_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init, dtype=torch.float32))

        sigma = torch.tensor(gaussian_sigma_init, dtype=torch.float32)
        if sigma.numel() < num_heads:
            repeats = (num_heads + sigma.numel() - 1) // sigma.numel()
            sigma = sigma.repeat(repeats)[:num_heads]
        self.log_sigma = nn.Parameter(torch.log(sigma))

    def _compute_position_bias(
        self, positions: Optional[Tensor], mask: Tensor
    ) -> Optional[Tensor]:
        if positions is None:
            return None
        pos_diff = positions.unsqueeze(-1) - positions.unsqueeze(-2)
        pos_sq = (pos_diff ** 2).unsqueeze(1)
        sigma = torch.exp(self.log_sigma).view(1, -1, 1, 1)
        bias = -pos_sq / (2.0 * sigma ** 2)
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            bias = bias.masked_fill(expanded_mask, 0.0)
        return self.bias_strength * bias

    def _inject_flag(self, tokens: Tensor, flags: Optional[Tensor], query_count: int) -> Tensor:
        if flags is None:
            return tokens
        if flags.dim() != 2:
            raise ValueError("strand_flag tensor must have shape (B, R)")
        indices = (flags.to(device=tokens.device) > 0).long().clamp(min=0, max=1)
        embedded = self.flag_embed(indices)
        expanded = _expand_batch_time(embedded, query_count)
        augmented = torch.cat([tokens, expanded], dim=-1)
        return self.flag_proj(augmented)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = dict(batch)
        if "fusion_embedding_per_read" not in batch:
            raise KeyError("Expected 'fusion_embedding_per_read' in batch")

        per_read = batch["fusion_embedding_per_read"]
        if per_read.dim() != 4:
            raise ValueError("Expected fusion_embedding_per_read to have shape (B, R, Q, H)")
        batch_size, num_reads, query_count, hidden_dim = per_read.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Aggregator configured for hidden_dim={self.hidden_dim}, received {hidden_dim}"
            )

        read_mask = batch.get("fusion_read_mask")
        if read_mask is None:
            raise KeyError("Expected 'fusion_read_mask' in batch for padding management")
        if read_mask.shape != (batch_size, num_reads):
            raise ValueError(
                f"fusion_read_mask must have shape ({batch_size}, {num_reads}),"
                f" received {tuple(read_mask.shape)}"
            )
        read_mask_bool = read_mask <= 0

        flags = batch.get("strand_flag")
        if flags is None:
            flags = batch.get("flag")

        flat_mask = _expand_batch_time(read_mask_bool.to(dtype=torch.bool), query_count)

        tokens = per_read.permute(0, 2, 1, 3).reshape(batch_size * query_count, num_reads, hidden_dim)
        tokens = self._inject_flag(tokens, flags, query_count)

        token_mask = flat_mask.unsqueeze(-1).to(device=tokens.device)
        tokens = tokens.masked_fill(token_mask, 0.0)

        positions = batch.get("fusion_soft_position")
        position_bias = None
        if positions is not None:
            if positions.shape != (batch_size, num_reads, query_count):
                raise ValueError(
                    "fusion_soft_position must have shape (B, R, Q) to match fusion outputs"
                )
            flat_positions = positions.permute(0, 2, 1).reshape(batch_size * query_count, num_reads)
            position_bias = self._compute_position_bias(flat_positions, flat_mask)

        encoded = tokens
        for block in self.blocks:
            encoded = block(encoded, key_padding_mask=flat_mask, bias=position_bias)
        encoded = encoded.contiguous()

        pooled, attn_weights = self.pma(encoded, key_padding_mask=flat_mask, need_weights=True)
        pooled_flat = pooled.reshape(batch_size * query_count, -1)
        projected = self.output_norm(self.k_proj(pooled_flat)).view(batch_size, query_count, hidden_dim)

        base = batch.get("embedding")
        if base is not None and base.shape != projected.shape:
            raise ValueError(
                f"Residual base has shape {tuple(base.shape)}, expected {tuple(projected.shape)}"
            )
        if base is None:
            aggregated = projected
        else:
            aggregated = base + self.residual_scale * projected
        batch["embedding"] = aggregated

        if attn_weights is not None:
            weights = attn_weights.mean(dim=1)
            weights = weights.view(batch_size, query_count, self.pma.num_seeds, num_reads)
            weights = weights.mean(dim=2)
            read_mask_expand = read_mask_bool.unsqueeze(1).expand(-1, query_count, -1)
            weights = weights.masked_fill(read_mask_expand, 0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            batch["aggregator_read_weights"] = weights

            if self.include_uncertainty:
                weight_flat = weights.view(batch_size * query_count, num_reads)
                value_flat = encoded.reshape(batch_size * query_count, num_reads, -1)
                value_flat = value_flat.masked_fill(token_mask.to(device=value_flat.device), 0.0)
                entropy = -(weight_flat * weight_flat.clamp_min(1e-6).log()).sum(dim=-1)
                mean = torch.matmul(weight_flat.unsqueeze(1), value_flat).squeeze(1)
                centered = value_flat - mean.unsqueeze(1)
                variance = (
                    (weight_flat.unsqueeze(-1) * centered.pow(2)).sum(dim=1).mean(dim=-1)
                )
                uncertainty = torch.stack([entropy, variance], dim=-1).view(
                    batch_size, query_count, 2
                )
                batch["aggregator_uncertainty"] = uncertainty

        return batch


@AGGREGATOR_REGISTRY.register("set_transformer")
def build_set_transformer_aggregator(**kwargs) -> AggregatorBase:
    return SetTransformerAggregator(**kwargs)


__all__ = ["SetTransformerAggregator", "build_set_transformer_aggregator"]