"""Dual-branch encoder that fuses signal and move/base hints."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .base import EncoderBase, ENCODER_REGISTRY


class DualBranchEncoder(EncoderBase):
    """Encode stride-windowed signals with move/base priors."""

    def __init__(
        self,
        signal_dim: int = 6,
        hidden_dim: int = 128,
        signal_kernel_size: int = 3,
        context_kernel_size: int = 5,
        signal_kernel_sizes: Sequence[int] | None = None,
        use_identity_residual: bool = False,
        signal_block: str = "multikernel",
        signal_block_params: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        signal_kernel_sizes = self._validate_signal_kernels(signal_kernel_size, signal_kernel_sizes)
        self.signal_kernel_sizes = list(signal_kernel_sizes)
        if context_kernel_size % 2 == 0 or context_kernel_size < 1:
            raise ValueError("context_kernel_size must be a positive odd integer")

        padding_context = context_kernel_size // 2

        self.use_identity_residual = use_identity_residual
        self.signal_norm = nn.LayerNorm(signal_dim)
        self.signal_block = signal_block.lower()
        block_params = dict(signal_block_params or {})
        self.signal_conv = self._build_signal_block(
            block=self.signal_block,
            signal_dim=signal_dim,
            hidden_dim=hidden_dim,
            kernel_sizes=self.signal_kernel_sizes,
            block_params=block_params,
            enable_residual=True,
        )
        if self.use_identity_residual:
            self.signal_skip_proj = nn.Linear(signal_dim, hidden_dim)
            self.signal_residual_norm = nn.LayerNorm(hidden_dim)
            residual_params = dict(signal_block_params or {})
            self.signal_residual_conv = self._build_signal_block(
                block=self.signal_block,
                signal_dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_sizes=self.signal_kernel_sizes,
                block_params=residual_params,
                enable_residual=False,
            )
        else:
            self.signal_skip_proj = None
            self.signal_residual_norm = None
            self.signal_residual_conv = None
        self.signal_out_norm = nn.LayerNorm(hidden_dim)

        self.move_embedding = nn.Embedding(2, hidden_dim)
        self.base_embedding = nn.Embedding(5, hidden_dim)
        self.prior_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=context_kernel_size, padding=padding_context, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.prior_norm = nn.LayerNorm(hidden_dim)
        self.prior_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hint_proj = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        signal = batch["signal"]
        move = batch["move"].long()
        base_index = batch["base_index"].long()

        if signal.dim() == 4:
            batch_size, num_reads, time_steps, feature_dim = signal.shape
            flat_signal = signal.view(batch_size * num_reads, time_steps, feature_dim)
            flat_move = move.view(batch_size * num_reads, time_steps)
            flat_base = base_index.view(batch_size * num_reads, time_steps)
            reshape_back = True
        elif signal.dim() == 3:
            flat_signal = signal
            flat_move = move
            flat_base = base_index
            reshape_back = False
            batch_size, time_steps, feature_dim = signal.shape
            num_reads = 1
        else:
            raise ValueError(
                "Signal tensor must have shape (B, T, F) or (B, R, T, F)"
            )

        if self.use_identity_residual:
            base_signal = self.signal_skip_proj(flat_signal)
            residual_input = self.signal_residual_norm(base_signal)
            signal_delta = self.signal_residual_conv(residual_input.transpose(1, 2)).transpose(1, 2)
            signal_delta = self.signal_out_norm(signal_delta)
            signal_feat = base_signal + signal_delta
        else:
            signal_norm = self.signal_norm(flat_signal)
            signal_feat = self.signal_conv(signal_norm.transpose(1, 2)).transpose(1, 2)
            signal_feat = self.signal_out_norm(signal_feat)

        move_embed = self.move_embedding(flat_move)
        base_embed = self.base_embedding(flat_base.clamp(min=0, max=4))
        prior_local = move_embed + base_embed
        prior_feat = self.prior_conv(prior_local.transpose(1, 2)).transpose(1, 2)
        prior_feat = self.prior_norm(prior_feat)

        soft_hint = torch.sigmoid(self.hint_proj(prior_feat))
        fused = signal_feat + soft_hint * self.prior_proj(prior_feat)

        if reshape_back:
            fused = fused.view(batch_size, num_reads, time_steps, -1)
            soft_hint = soft_hint.view(batch_size, num_reads, time_steps)
        else:
            fused = fused
            soft_hint = soft_hint.squeeze(-1)

        batch = dict(batch)
        batch["embedding"] = fused
        hard_mask = batch.get("hard_mask")
        if hard_mask is None:
            hard_mask = batch["move"].float()
        batch["hard_mask"] = hard_mask
        batch["soft_hint"] = soft_hint
        return batch

    @staticmethod
    def _validate_signal_kernels(
        signal_kernel_size: int,
        signal_kernel_sizes: Sequence[int] | None,
    ) -> List[int]:
        if signal_kernel_sizes is None:
            kernel_sizes: Iterable[int] = (signal_kernel_size,)
        else:
            if isinstance(signal_kernel_sizes, Iterable) and not isinstance(signal_kernel_sizes, (str, bytes)):
                kernel_sizes = signal_kernel_sizes
            else:
                raise TypeError("signal_kernel_sizes must be a sequence of positive odd integers")

        validated: List[int] = []
        for kernel in kernel_sizes:
            if not isinstance(kernel, int):
                raise TypeError("signal kernel sizes must be integers")
            if kernel % 2 == 0 or kernel < 1:
                raise ValueError("signal kernel sizes must be positive odd integers")
            validated.append(kernel)

        if not validated:
            raise ValueError("at least one signal kernel size must be provided")

        return validated

    def _build_signal_block(
        self,
        block: str,
        signal_dim: int,
        hidden_dim: int,
        kernel_sizes: Sequence[int],
        block_params: Dict[str, Any],
        *,
        enable_residual: bool,
    ) -> nn.Module:
        if block == "multikernel":
            return MultiKernelSignalConv(
                signal_dim=signal_dim,
                hidden_dim=hidden_dim,
                kernel_sizes=kernel_sizes,
                **block_params,
            )
        if block == "inception_lite":
            inception_defaults: Dict[str, Any] = {
                "bottleneck": None,
                "kernel_sizes": (3, 5, 9, 15),
                "dilations": (1, 1, 2, 1),
                "pool_kernel_size": 3,
                "gn_groups": 32,
                "drop_path": 0.0,
                "use_se": False,
                "se_reduction": 16,
                "use_residual": enable_residual,
            }
            inception_defaults.update(block_params)
            inception_defaults["use_residual"] = enable_residual
            return Inception1DLite(
                in_channels=signal_dim,
                out_channels=hidden_dim,
                **inception_defaults,
            )
        raise ValueError(f"Unsupported signal_block '{block}'. Expected 'multikernel' or 'inception_lite'.")


class SelectiveKernel1D(nn.Module):
    """Selective kernel aggregation for 1D feature maps."""

    def __init__(
        self,
        channels: int,
        num_branches: int,
        reduction: int = 16,
        min_channels: int = 8,
    ) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, min_channels)
        self.fc1 = nn.Conv1d(channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(reduced_channels, channels * num_branches, kernel_size=1)
        self.activation = nn.SiLU()
        self.num_branches = num_branches

    def forward(self, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(branches) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branch tensors, but received {len(branches)}"
            )
        if len(branches) == 0:
            raise ValueError("branches must contain at least one tensor")

        aggregated = torch.stack(branches, dim=0).sum(dim=0)
        squeeze = F.adaptive_avg_pool1d(aggregated, 1)
        excitation = self.activation(self.fc1(squeeze))
        attention = self.fc2(excitation)
        batch_size, channels, _ = aggregated.shape
        attention = attention.view(batch_size, self.num_branches, channels, 1)
        attention = attention.softmax(dim=1)

        weighted = []
        for idx, branch in enumerate(branches):
            weight = attention[:, idx]
            weighted.append(weight * branch)
        return torch.stack(weighted, dim=0).sum(dim=0)


class MultiKernelSignalConv(nn.Module):
    """Signal convolution block that supports multi-kernel feature extraction."""

    def __init__(
        self,
        signal_dim: int,
        hidden_dim: int,
        kernel_sizes: Sequence[int],
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.pre = nn.Sequential(
            nn.Conv1d(signal_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.SiLU(),
                )
                for k in self.kernel_sizes
            ]
        )
        self.selector = (
            SelectiveKernel1D(hidden_dim, len(self.kernel_sizes), reduction=reduction)
            if len(self.kernel_sizes) > 1
            else None
        )
        self.post = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        branch_outputs = [branch(x) for branch in self.branches]
        if self.selector is None:
            combined = branch_outputs[0]
        else:
            combined = self.selector(branch_outputs)
        return self.post(combined)


class SqueezeExcite1D(nn.Module):
    """Lightweight squeeze-and-excitation block for 1D signals."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool1d(x, 1)
        scale = self.activation(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor = random_tensor.div(keep_prob)
    return x * random_tensor


class Inception1DLite(nn.Module):
    """Depthwise-separable Inception-style block for 1D signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck: int | None = None,
        kernel_sizes: Sequence[int] = (3, 5, 9, 15),
        dilations: Sequence[int] = (1, 1, 2, 1),
        pool_kernel_size: int = 3,
        gn_groups: int = 32,
        drop_path: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if pool_kernel_size % 2 == 0 or pool_kernel_size < 1:
            raise ValueError("pool_kernel_size must be a positive odd integer")
        if len(kernel_sizes) < 2:
            raise ValueError("kernel_sizes must provide at least two values for depthwise branches")
        if len(dilations) < 1:
            raise ValueError("dilations must contain at least one value")

        bottleneck_dim = bottleneck or max(out_channels // 4, 1)
        self.drop_path_prob = float(drop_path)
        small_kernel = kernel_sizes[0]
        small_dilation = dilations[0]
        big_kernel = kernel_sizes[-1]
        big_dilation = dilations[-1]
        self.branch_identity = nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1)
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1),
            nn.Conv1d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=small_kernel,
                padding=self._calc_padding(small_kernel, small_dilation),
                groups=bottleneck_dim,
                dilation=small_dilation,
            ),
            nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=1),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1),
            nn.Conv1d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=big_kernel,
                padding=self._calc_padding(big_kernel, big_dilation),
                groups=bottleneck_dim,
                dilation=big_dilation,
            ),
            nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=1),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False),
            nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1),
        )
        fused_channels = bottleneck_dim * 4
        self.fusion = nn.Conv1d(fused_channels, out_channels * 2, kernel_size=1)
        norm_groups = gn_groups if out_channels % gn_groups == 0 else math.gcd(out_channels, gn_groups)
        norm_groups = max(norm_groups, 1)
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.use_se = use_se
        self.se = SqueezeExcite1D(out_channels, reduction=se_reduction) if use_se else None
        self.use_residual = use_residual
        if self.use_residual:
            self.residual = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )
        else:
            self.residual = None

    @staticmethod
    def _calc_padding(kernel_size: int, dilation: int) -> int:
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd integer")
        return (kernel_size - 1) // 2 * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            self.branch_identity(x),
            self.branch_small(x),
            self.branch_large(x),
            self.branch_pool(x),
        ]
        fused = torch.cat(branches, dim=1)
        fused = F.glu(self.fusion(fused), dim=1)
        fused = self.norm(fused)
        if self.se is not None:
            fused = self.se(fused)
        fused = _drop_path(fused, self.drop_path_prob, self.training)
        if self.use_residual:
            identity = self.residual(x)
            return fused + identity
        return fused


@ENCODER_REGISTRY.register("dual_branch")
def build_dual_branch_encoder(**kwargs: Any) -> EncoderBase:
    return DualBranchEncoder(**kwargs)


__all__ = [
    "DualBranchEncoder",
    "build_dual_branch_encoder",
    "Inception1DLite",
    "MultiKernelSignalConv",
]