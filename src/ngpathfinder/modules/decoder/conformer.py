"""Conformer-based temporal encoder and CTC decoder."""
from __future__ import annotations

import math
from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from ..encoder.dual_branch import Inception1DLite
from .base import DECODER_REGISTRY, DecoderBase
from .ctc import CTCDecoder
from .ctc_crf import _build_alibi_bias, TemporalMultiHeadAttention


class Swish(nn.Module):
    """Memory-efficient SiLU/Swish activation."""

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - simple wrapper
        return x * torch.sigmoid(x)


class ConformerFeedForward(nn.Module):
    """Macaron-style feed-forward module with residual scaling."""

    def __init__(self, hidden_dim: int, expansion_factor: float, dropout: float) -> None:
        super().__init__()
        if expansion_factor < 1.0:
            raise ValueError("expansion_factor must be >= 1.0")
        inner_dim = int(hidden_dim * expansion_factor)
        if inner_dim < hidden_dim:
            inner_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.scale = 0.5

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.net(x)


class ConformerConvModule(nn.Module):
    """Depth-wise separable convolutional module used in Conformer blocks."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        expansion_factor: float,
        dropout: float,
        *,
        use_inception: bool = False,
        inception_params: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size must be odd and >= 3")
        if expansion_factor < 1.0:
            raise ValueError("expansion_factor must be >= 1.0")

        expanded_dim = int(hidden_dim * expansion_factor)
        if expanded_dim < hidden_dim:
            expanded_dim = hidden_dim

        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.pointwise_in = nn.Conv1d(hidden_dim, 2 * expanded_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.use_inception = bool(use_inception)
        if self.use_inception:
            extra_args = dict(inception_params or {})
            self.inception = Inception1DLite(
                in_channels=expanded_dim,
                out_channels=expanded_dim,
                use_residual=False,
                **extra_args,
            )
            self.depthwise = None
            self.norm = None
            self.activation = None
        else:
            self.inception = None
            self.depthwise = nn.Conv1d(
                expanded_dim,
                expanded_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=expanded_dim,
            )
            self.norm = nn.BatchNorm1d(expanded_dim)
            self.activation = Swish()
        self.pointwise_out = nn.Conv1d(expanded_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        # x: (batch, time, dim) -> (batch, dim, time)
        residual = x
        x = self.pre_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        x = self.glu(x)
        if self.use_inception:
            if self.inception is None:
                raise RuntimeError("Inception module is not initialized")
            x = self.inception(x)
        else:
            if self.depthwise is None or self.norm is None or self.activation is None:
                raise RuntimeError("Depthwise convolution components are not initialized")
            x = self.depthwise(x)
            x = self.norm(x)
            x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        if key_padding_mask is not None:
            mask = (~key_padding_mask).to(dtype=x.dtype).unsqueeze(-1)
            x = x * mask
        return residual + x


class ConformerBlock(nn.Module):
    """Standard Conformer block (FFN-MHSA-Conv-FFN with Pre-LN)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
        conv_kernel_size: int,
        conv_expansion_factor: float,
        conv_dropout: float,
        *,
        conv_use_inception: bool = False,
        conv_inception_params: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(hidden_dim, ffn_multiplier, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = TemporalMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(
            hidden_dim,
            conv_kernel_size,
            conv_expansion_factor,
            conv_dropout,
            use_inception=conv_use_inception,
            inception_params=conv_inception_params,
        )
        self.ffn2 = ConformerFeedForward(hidden_dim, ffn_multiplier, dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        *,
        attention_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.ffn1(x)

        residual = x
        attn_input = self.attn_norm(x)
        attn_output = self.attn(
            attn_input, attention_bias=attention_bias, key_padding_mask=key_padding_mask
        )
        x = residual + self.attn_dropout(attn_output)

        x = self.conv_module(x, key_padding_mask)
        x = self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks with shared ALiBi bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
        *,
        conv_kernel_size: int = 31,
        conv_expansion_factor: float = 2.0,
        conv_dropout: Optional[float] = None,
        conv_use_inception: bool = False,
        conv_inception_params: Optional[Mapping[str, object]] = None,
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if conv_dropout is None:
            conv_dropout = dropout
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_dim,
                    num_heads,
                    dropout,
                    ffn_multiplier,
                    conv_kernel_size,
                    conv_expansion_factor,
                    conv_dropout,
                    conv_use_inception=conv_use_inception,
                    conv_inception_params=conv_inception_params,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
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
            return self.final_norm(x)

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


class CTCConformerDecoder(CTCDecoder):
    """CTC decoder backed by a Conformer encoder stack."""

    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: float = 4.0,
        *,
        conv_kernel_size: int = 31,
        conv_expansion_factor: float = 2.0,
        conv_dropout: Optional[float] = None,
        conv_use_inception: bool = False,
        conv_inception_params: Optional[Mapping[str, object]] = None,
        use_temporal_positional_encoding: bool = False,
        enable_duration_head: bool = False,
        duration_head_hidden_scale: float = 1.0,
    ) -> None:
        super().__init__(
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            use_temporal_positional_encoding=use_temporal_positional_encoding,
            enable_duration_head=enable_duration_head,
            duration_head_hidden_scale=duration_head_hidden_scale,
        )
        conv_dropout = dropout if conv_dropout is None else float(conv_dropout)
        self.transformer = ConformerEncoder(
            hidden_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            conv_kernel_size=conv_kernel_size,
            conv_expansion_factor=conv_expansion_factor,
            conv_dropout=conv_dropout,
            conv_use_inception=conv_use_inception,
            conv_inception_params=conv_inception_params,
            use_positional_encoding=use_temporal_positional_encoding,
        )


@DECODER_REGISTRY.register("ctc_conformer")
def build_ctc_conformer_decoder(**kwargs: object) -> DecoderBase:
    return CTCConformerDecoder(**kwargs)


__all__ = [
    "ConformerBlock",
    "ConformerEncoder",
    "CTCConformerDecoder",
    "build_ctc_conformer_decoder",
]