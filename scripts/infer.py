"""Inference entry point for NanoGraph PathFinder2."""
from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass, field
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import ComponentConfig, load_config
from ngpathfinder.data import ReadDataset, collate_read_batch, list_block_segments
try:
    from ngpathfinder.losses.ctc_crf import NUM_BASES as CTC_NUM_BASES
    from ngpathfinder.losses.ctc_crf import _gather_reference
except ImportError:  # pragma: no cover - optional dependency fallback
    from ngpathfinder.losses.ctc_constants import NUM_BASES as CTC_NUM_BASES
    from ngpathfinder.losses.ctc_common import _gather_reference
from ngpathfinder.modules.aggregator import AGGREGATOR_REGISTRY
from ngpathfinder.modules.decoder import DECODER_REGISTRY
from ngpathfinder.modules.decoder.rnnt import RNNTDecoder
from ngpathfinder.modules.encoder import ENCODER_REGISTRY
from ngpathfinder.modules.fusion import FUSION_REGISTRY
from ngpathfinder.utils.checkpoint import load_checkpoint
from ngpathfinder.utils.logging import configure_logging

BASE_VOCAB = {1: "A", 2: "C", 3: "G", 4: "T"}


def _component_kwargs(component: ComponentConfig) -> Dict[str, Any]:
    params = dict(component.params)
    if component.variant:
        params.setdefault("variant", component.variant)
    return params


@dataclass(frozen=True)
class DecodeOptions:
    """Container for decoding strategy selections."""

    strategy: str = "greedy"
    beam_width: int = 10
    beam_prune_threshold: Optional[float] = None
    torchaudio_params: Dict[str, Any] = field(default_factory=dict)
    use_duration_prior: bool = False
    rnnt_max_symbols_per_step: int = 32
    rnnt_strategy: str = "greedy"


@dataclass(frozen=True)
class ViterbiDecodeResult:
    """Decoded token streams derived from the CTC-CRF Viterbi path."""

    collapsed: str
    expanded: str
    collapsed_length: int
    expanded_length: int
    run_bases: List[int] = field(default_factory=list)
    char_run_lengths: List[int] = field(default_factory=list)
    frame_run_lengths: List[int] = field(default_factory=list)
    collapsed_tokens: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class DecodedRecord:
    """Book-keeping structure for decoded sequences and summary stats."""

    segment: str
    sequence: str  # Collapsed sequence used for metric computation.
    logit_length: int
    non_blank: int
    run_total: int
    frame_run_total: Optional[int] = None
    same_base_reenter_count: Optional[int] = None
    collapsed_sequence: Optional[str] = None
    collapsed_length: Optional[int] = None
    expanded_sequence: Optional[str] = None
    expanded_length: Optional[int] = None


@dataclass(frozen=True)
class AlignmentStats:
    """Alignment summary with optional traceback metadata."""

    distance: int
    alignment_length: int
    matches: int
    reference_length: int
    operations: List[str] = field(default_factory=list)
    reference_indices: List[Optional[int]] = field(default_factory=list)


def _dataset_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    use_fasta = params.get("use_fasta_reference", True)
    if not use_fasta:
        raise ValueError(
            "ReadDataset now requires FASTA references; set data.params.use_fasta_reference to true"
        )
    dataset_type = str(params.get("type", "legacy")).lower()
    return {
        "dataset_type": dataset_type,
        "max_mv_len": params.get("max_mv_len"),
        "max_reads_per_segment": params.get("max_reads_per_segment"),
        "fasta_glob_patterns": params.get("fasta_glob_patterns"),
        "ambiguous_base_policy": params.get("ambiguous_base_policy", "error"),
        "fasta_sequence_policy": params.get("fasta_sequence_policy", "first"),
        "suppress_mv_len_warnings": params.get("suppress_mv_len_warnings", False),
        "use_fastq_base_sequence": params.get("use_fastq_base_sequence", True),
        "use_read_flag": params.get("use_read_flag", True),
    }


def _resolve_path(path_str: str | None) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _split_segments(
    segment_names: Sequence[str], split_config: Mapping[str, Any], seed: int
) -> Tuple[List[str], List[str], List[str]]:
    if not segment_names:
        raise ValueError("No segments available to split")

    train_ratio = float(split_config.get("train", 0.8))
    val_ratio = float(split_config.get("val", 0.1))
    test_ratio = float(split_config.get("test", max(0.0, 1.0 - train_ratio - val_ratio)))

    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    if total_ratio > 1.0 + 1e-6:
        raise ValueError("Split ratios may not sum to more than 1")

    names = list(segment_names)
    names.sort()
    rng = random.Random(seed)
    rng.shuffle(names)

    normalized = [train_ratio, val_ratio, test_ratio]
    ratio_sum = sum(normalized)
    normalized = [value / ratio_sum for value in normalized]

    counts = [int(len(names) * value) for value in normalized]
    while sum(counts) < len(names):
        for idx in range(len(counts)):
            if sum(counts) >= len(names):
                break
            counts[idx] += 1

    train_count, val_count, test_count = counts
    if train_count <= 0:
        raise ValueError("Train split must contain at least one segment")

    train = names[:train_count]
    val = names[train_count : train_count + val_count]
    test = names[train_count + val_count : train_count + val_count + test_count]
    return train, val, test


def _log_sum_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _compact_padding_mask(mask: torch.Tensor, target_steps: int) -> torch.Tensor:
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
        mask = mask.to(torch.bool)
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
        return mask.to(torch.bool)
    raise ValueError("decoder_padding_mask must have shape (B, T) or (B, R, T)")


def _extract_valid_ctc_logits(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    gather_indices: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gather_indices is not None:
        if gather_indices.dim() != 2:
            raise ValueError("ctc_logit_gather_indices must have shape (B, T)")
        if gather_indices.size(0) != logits.size(0):
            raise ValueError("ctc_logit_gather_indices must align with logits batch dimension")
        gather_indices = gather_indices.to(device=logits.device, dtype=torch.long)
        batch, max_valid = gather_indices.shape
        vocab = logits.size(-1)
        # ``ctc_logits`` emitted by the decoder are already compacted; only gather when the
        # logits still contain padded time steps from the original sequence length.
        if logits.size(1) > max_valid:
            compact = logits.new_zeros((batch, max_valid, vocab))
            for batch_idx in range(batch):
                valid_len = int(lengths[batch_idx].item()) if batch_idx < lengths.numel() else 0
                if valid_len <= 0:
                    continue
                step_indices = gather_indices[batch_idx, :valid_len]
                compact[batch_idx, :valid_len] = logits[batch_idx].index_select(0, step_indices)
            return compact

        if logits.size(1) < max_valid:
            compact = logits.new_zeros((batch, max_valid, vocab))
            compact[:, : logits.size(1)] = logits
            return compact

        return logits[:, :max_valid]

    if padding_mask is not None:
        normalized_mask = _compact_padding_mask(padding_mask, logits.size(1))
        valid_mask = ~normalized_mask
        inferred_lengths = valid_mask.sum(dim=1)
        max_valid = int(inferred_lengths.max().item()) if inferred_lengths.numel() > 0 else 0
        vocab = logits.size(-1)
        compact = logits.new_zeros((logits.size(0), max_valid, vocab))
        for batch_idx in range(logits.size(0)):
            valid_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).view(-1)
            if valid_indices.numel() == 0:
                continue
            compact[batch_idx, : valid_indices.numel()] = logits[batch_idx].index_select(0, valid_indices)
        return compact

    return logits


def _extract_valid_duration_logits(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    gather_indices: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gather_indices is not None:
        if gather_indices.dim() != 2:
            raise ValueError("ctc_logit_gather_indices must have shape (B, T)")
        gather_indices = gather_indices.to(device=logits.device, dtype=torch.long)
        batch, max_valid = gather_indices.shape
        if logits.size(1) == max_valid:
            return logits[:, :max_valid]
        compact = logits.new_zeros((batch, max_valid))
        for batch_idx in range(batch):
            valid_len = int(lengths[batch_idx].item()) if batch_idx < lengths.numel() else 0
            if valid_len <= 0:
                continue
            step_indices = gather_indices[batch_idx, :valid_len]
            compact[batch_idx, :valid_len] = logits[batch_idx].index_select(0, step_indices)
        return compact

    if padding_mask is not None:
        normalized_mask = _compact_padding_mask(padding_mask, logits.size(1))
        valid_mask = ~normalized_mask
        inferred_lengths = valid_mask.sum(dim=1)
        max_valid = int(inferred_lengths.max().item()) if inferred_lengths.numel() > 0 else 0
        compact = logits.new_zeros((logits.size(0), max_valid))
        for batch_idx in range(logits.size(0)):
            valid_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).view(-1)
            if valid_indices.numel() == 0:
                continue
            compact[batch_idx, : valid_indices.numel()] = logits[batch_idx].index_select(
                0, valid_indices
            )
        return compact

    return logits


def _decode_viterbi_sequence(
    run_bases: List[int],
    *,
    char_run_lengths: Optional[List[int]] = None,
    frame_run_lengths: Optional[List[int]] = None,
    collapsed_tokens: Optional[List[int]] = None,
) -> ViterbiDecodeResult:
    """Decode a Viterbi token stream into collapsed and expanded strings."""

    def _to_int_list(values: Optional[List[int]]) -> List[int]:
        output: List[int] = []
        if not values:
            return output
        for value in values:
            if isinstance(value, torch.Tensor):
                output.append(int(value.item()))
            else:
                output.append(int(value))
        return output

    run_bases_int = _to_int_list(run_bases)
    char_run_lengths_int = _to_int_list(char_run_lengths)
    frame_run_lengths_int = _to_int_list(frame_run_lengths)
    collapsed_token_int = _to_int_list(collapsed_tokens)

    if not char_run_lengths_int and run_bases_int:
        char_run_lengths_int = [1 for _ in run_bases_int]

    if not run_bases_int and collapsed_token_int:
        deduped: List[int] = []
        last_token: Optional[int] = None
        for token in collapsed_token_int:
            token_int = int(token)
            if last_token is None or token_int != last_token:
                deduped.append(token_int)
                last_token = token_int
        run_bases_int = deduped
        if not char_run_lengths_int:
            char_run_lengths_int = [1 for _ in run_bases_int]

    if not collapsed_token_int:
        collapsed_token_int = []
        for base, length in zip(run_bases_int, char_run_lengths_int):
            collapsed_token_int.extend([base] * max(length, 1))

    run_count = len(run_bases_int)
    if run_count and len(char_run_lengths_int) < run_count:
        char_run_lengths_int = char_run_lengths_int + [1] * (run_count - len(char_run_lengths_int))
    elif run_count and len(char_run_lengths_int) > run_count:
        char_run_lengths_int = char_run_lengths_int[:run_count]

    if not frame_run_lengths_int:
        frame_run_lengths_int = [max(length, 1) for length in char_run_lengths_int]
    elif run_count and len(frame_run_lengths_int) < run_count:
        frame_run_lengths_int = frame_run_lengths_int + [1] * (run_count - len(frame_run_lengths_int))
    elif run_count and len(frame_run_lengths_int) > run_count:
        frame_run_lengths_int = frame_run_lengths_int[:run_count]

    collapsed_parts: List[str] = [
        BASE_VOCAB.get(int(base), "N") for base in run_bases_int
    ]

    expanded_parts: List[str] = [
        BASE_VOCAB.get(int(base), "N") * max(run, 1)
        for base, run in zip(run_bases_int, char_run_lengths_int)
    ]

    collapsed_string = "".join(collapsed_parts)
    expanded_string = "".join(expanded_parts)

    return ViterbiDecodeResult(
        collapsed=collapsed_string,
        expanded=expanded_string,
        collapsed_length=len(collapsed_parts),
        expanded_length=sum(max(r, 1) for r in char_run_lengths_int[: run_count or None]),
        run_bases=run_bases_int,
        char_run_lengths=char_run_lengths_int,
        frame_run_lengths=frame_run_lengths_int,
        collapsed_tokens=collapsed_token_int,
    )


def _summarize_viterbi_stats(result: ViterbiDecodeResult) -> Dict[str, float]:
    char_total = sum(max(length, 1) for length in result.char_run_lengths)
    frame_total = sum(max(length, 1) for length in result.frame_run_lengths)
    same_base_reenter = sum(max(length - 1, 0) for length in result.char_run_lengths)
    enter_count = char_total
    extend_count = max(frame_total - char_total, 0)
    enter_rate = enter_count / float(frame_total) if frame_total > 0 else 0.0
    return {
        "enter_count": float(enter_count),
        "extend_count": float(extend_count),
        "enter_rate": enter_rate,
        "char_total": float(char_total),
        "frame_total": float(frame_total),
        "collapsed_length": float(result.collapsed_length),
        "same_base_reenter": float(same_base_reenter),
    }


def _log_viterbi_stats(logger: logging.Logger, segment: str, result: ViterbiDecodeResult) -> Dict[str, float]:
    stats = _summarize_viterbi_stats(result)
    logger.info(
        (
            "Viterbi diagnostics | segment=%s | T_eff=%d | #enter=%d | #extend=%d | "
            "enter_rate=%.4f | len(collapsed)=%d | sum_char_rle=%d | sum_frame_rle=%d | "
            "same_base_reenter=%d"
        ),
        segment,
        int(stats["frame_total"]),
        int(stats["enter_count"]),
        int(stats["extend_count"]),
        stats["enter_rate"],
        int(stats["collapsed_length"]),
        int(stats["char_total"]),
        int(stats["frame_total"]),
        int(stats["same_base_reenter"]),
    )
    return stats


def _select_sequence_item(source: Any, idx: int) -> Optional[List[int]]:
    if source is None:
        return None
    if isinstance(source, list):
        if idx >= len(source):
            return None
        item = source[idx]
        if item is None:
            return None
        if torch.is_tensor(item):
            return item.detach().to("cpu").to(torch.int64).tolist()
        if isinstance(item, (list, tuple)):
            return [int(val) for val in item]
        return [int(item)]
    if torch.is_tensor(source):
        tensor = source
        if tensor.dim() == 0:
            return [int(tensor.item())]
        if tensor.dim() == 1:
            return tensor.detach().to("cpu").to(torch.int64).tolist()
        if idx < tensor.size(0):
            return tensor[idx].detach().to("cpu").to(torch.int64).tolist()
    return None


def _ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    valid_length: int,
    beam_width: int,
    blank_index: int = 0,
    prune_threshold: Optional[float] = None,
    duration_priors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> str:
    """Prefix beam search decoder for CTC logits.

    Args:
        log_probs: Tensor of shape (time, vocab) containing log probabilities.
        valid_length: Number of time steps to consider from ``log_probs``.
        beam_width: Beam size to keep after each step.
        blank_index: Vocabulary index reserved for the blank token.
        prune_threshold: Optional per-step pruning threshold (in log prob space).

    Returns:
        Decoded base string produced by the highest probability prefix.
    """

    if valid_length <= 0 or log_probs.numel() == 0:
        return ""

    max_time = min(valid_length, log_probs.size(0))
    beam: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}

    stay_prior: Optional[torch.Tensor] = None
    advance_prior: Optional[torch.Tensor] = None
    if duration_priors is not None:
        stay_prior, advance_prior = duration_priors
        if stay_prior.size(0) < max_time or advance_prior.size(0) < max_time:
            raise ValueError("duration priors must have at least 'valid_length' entries")

    for t in range(max_time):
        step = log_probs[t]
        vocab_size = step.size(-1)

        stay_log = 0.0
        advance_log = 0.0
        if stay_prior is not None and advance_prior is not None:
            stay_log = float(stay_prior[t].item())
            advance_log = float(advance_prior[t].item())

        if prune_threshold is not None and prune_threshold > 0:
            mask = (step.max() - step) <= prune_threshold
            candidate_indices = mask.nonzero(as_tuple=False).view(-1).tolist()
        else:
            top_k = min(vocab_size, max(beam_width * 2, 1))
            candidate_indices = torch.topk(step, k=top_k).indices.tolist()

        if not candidate_indices:
            candidate_indices = list(range(vocab_size))

        candidate_set = set(candidate_indices)
        candidate_set.add(blank_index)

        next_beam: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        blank_log_prob = float(step[blank_index].item())

        for prefix, (prob_blank, prob_non_blank) in beam.items():
            total = _log_sum_exp(prob_blank, prob_non_blank)
            existing_blank, existing_non_blank = next_beam.get(prefix, (-math.inf, -math.inf))
            updated_blank = _log_sum_exp(existing_blank, blank_log_prob + total)
            next_beam[prefix] = (updated_blank, existing_non_blank)

        for prefix, (prob_blank, prob_non_blank) in beam.items():
            last_token = prefix[-1] if prefix else None
            total = _log_sum_exp(prob_blank, prob_non_blank)
            for token in candidate_set:
                if token == blank_index:
                    continue
                log_prob = float(step[token].item())
                if log_prob == -math.inf:
                    continue

                if token == last_token:
                    existing_blank, existing_non_blank = next_beam.get(prefix, (-math.inf, -math.inf))
                    updated_non_blank = _log_sum_exp(
                        existing_non_blank, log_prob + prob_non_blank + stay_log
                    )
                    next_beam[prefix] = (existing_blank, updated_non_blank)

                    new_prefix = prefix + (token,)
                    existing_blank, existing_non_blank = next_beam.get(new_prefix, (-math.inf, -math.inf))
                    updated_non_blank = _log_sum_exp(
                        existing_non_blank, log_prob + prob_blank + advance_log
                    )
                    next_beam[new_prefix] = (existing_blank, updated_non_blank)
                else:
                    new_prefix = prefix + (token,)
                    existing_blank, existing_non_blank = next_beam.get(new_prefix, (-math.inf, -math.inf))
                    updated_non_blank = _log_sum_exp(
                        existing_non_blank, log_prob + total + advance_log
                    )
                    next_beam[new_prefix] = (existing_blank, updated_non_blank)

        sorted_beam = sorted(
            next_beam.items(),
            key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
            reverse=True,
        )
        beam = dict(sorted_beam[: max(beam_width, 1)])

        if not beam:
            beam = {(): (-math.inf, -math.inf)}

    best_prefix = max(
        beam.items(), key=lambda item: _log_sum_exp(item[1][0], item[1][1])
    )[0]
    return "".join(BASE_VOCAB.get(token, "N") for token in best_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with NanoGraph PathFinder2 models")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--ckpt",
        default="",
        help="Checkpoint path for model weights. Overrides config defaults when provided.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to run inference on (default: test)",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=("greedy", "beam", "torchaudio", "viterbi"),
        default=None,
        help="Override decoding strategy (default: use config value or greedy)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Beam width when using beam decoding (default: use config or 10)",
    )
    parser.add_argument(
        "--beam-prune-threshold",
        type=float,
        default=None,
        help="Optional per-step log-probability pruning threshold for beam decoding",
    )
    parser.add_argument(
        "--torchaudio-nbest",
        type=int,
        default=None,
        help="Override number of hypotheses to retain when using torchaudio decoding",
    )
    parser.add_argument(
        "--torchaudio-beam-threshold",
        type=float,
        default=None,
        help="Override torchaudio beam threshold (score difference pruning)",
    )
    parser.add_argument(
        "--homopolymer-eval",
        action="store_true",
        help="Compute an additional set of metrics restricted to homopolymer regions",
    )
    parser.add_argument(
        "--homopolymer-threshold",
        type=int,
        default=8,
        help="Minimum run length (in reference bases) to classify a region as homopolymer",
    )
    parser.add_argument(
        "--homopolymer-mode",
        choices=("threshold", "center"),
        default="threshold",
        help=(
            "Strategy used to identify homopolymer regions when --homopolymer-eval is enabled. "
            "'threshold' selects all reference runs with length >= --homopolymer-threshold, "
            "while 'center' selects the run that contains the midpoint of the reference sequence."
        ),
    )
    return parser.parse_args()


def _sanitize_torchaudio_params(
    params: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """Validate torchaudio decoder overrides without mutating inputs."""

    sanitized: Dict[str, Any] = {}

    for key, value in params.items():
        if value is None:
            continue

        if key in {"nbest", "beam_size_token"}:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                logger.warning("Invalid torchaudio %s '%s'; ignoring override", key, value)
                continue
            if parsed <= 0:
                logger.warning("Torchaudio %s must be positive; ignoring value %d", key, parsed)
                continue
            sanitized[key] = parsed
        elif key in {"beam_threshold", "lm_weight", "word_score", "unk_score", "sil_score"}:
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid torchaudio %s '%s'; ignoring override", key, value)
        elif key == "log_add":
            sanitized[key] = bool(value)
        elif key in {"blank_token", "sil_token", "unk_word", "lexicon", "lm_dict"}:
            sanitized[key] = str(value)
        elif key == "lm":
            sanitized[key] = value
        else:
            logger.warning("Unknown torchaudio decoder option '%s'; ignoring", key)

    if "nbest" not in sanitized and params:
        sanitized["nbest"] = 1
    elif "nbest" in sanitized and sanitized["nbest"] <= 0:
        logger.warning("Torchaudio nbest must be positive; defaulting to 1")
        sanitized["nbest"] = 1

    return sanitized


def _resolve_decode_options(
    cfg: Any,
    logger: logging.Logger,
    strategy_override: Optional[str] = None,
    beam_width_override: Optional[int] = None,
    prune_override: Optional[float] = None,
    torchaudio_overrides: Optional[Dict[str, Any]] = None,
) -> DecodeOptions:
    inference_cfg = getattr(cfg, "inference", None)

    strategy_source = strategy_override or getattr(inference_cfg, "decode_strategy", "greedy")
    strategy = (strategy_source or "greedy").lower()
    valid_strategies = {"greedy", "beam", "torchaudio", "viterbi", "rnnt_greedy", "rnnt_beam"}
    if strategy not in valid_strategies:
        logger.warning("Unknown decode strategy '%s'; falling back to greedy decoding", strategy_source)
        strategy = "greedy"
    rnnt_strategy = "greedy"
    if strategy in {"rnnt_greedy", "rnnt_beam"}:
        rnnt_strategy = "beam" if strategy.endswith("beam") else "greedy"
        strategy = "greedy"

    beam_width_value: Optional[int] = beam_width_override
    if beam_width_value is None and inference_cfg is not None:
        beam_width_value = getattr(inference_cfg, "beam_width", None)
    if beam_width_value is None:
        beam_width_value = 10
    try:
        beam_width = int(beam_width_value)
    except (TypeError, ValueError):
        logger.warning("Invalid beam width '%s'; defaulting to 10", beam_width_value)
        beam_width = 10
    if beam_width <= 0:
        logger.warning("Beam width %d is not positive; defaulting to 10", beam_width)
        beam_width = 10

    prune_value = prune_override
    if prune_value is None and inference_cfg is not None:
        prune_value = getattr(inference_cfg, "beam_prune_threshold", None)
    prune_threshold: Optional[float]
    if prune_value is None:
        prune_threshold = None
    else:
        try:
            prune_threshold = float(prune_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid beam prune threshold '%s'; disabling per-step pruning", prune_value
            )
            prune_threshold = None
        else:
            if prune_threshold < 0:
                logger.warning(
                    "Beam prune threshold %.3f is negative; disabling per-step pruning",
                    prune_threshold,
                )
                prune_threshold = None

    torchaudio_params: Dict[str, Any] = {}
    if inference_cfg is not None:
        raw_torchaudio = getattr(inference_cfg, "torchaudio", {}) or {}
        if isinstance(raw_torchaudio, dict):
            torchaudio_params.update(raw_torchaudio)
        else:
            logger.warning(
                "Inference torchaudio configuration must be a mapping; ignoring value of type %s",
                type(raw_torchaudio).__name__,
            )

    if torchaudio_overrides:
        torchaudio_params.update(torchaudio_overrides)

    sanitized_torchaudio = _sanitize_torchaudio_params(torchaudio_params, logger)

    use_duration_prior = False
    if inference_cfg is not None:
        use_duration_prior = bool(getattr(inference_cfg, "use_duration_prior", False))

    rnnt_max_symbols = 32
    if inference_cfg is not None:
        rnnt_max_symbols = int(getattr(inference_cfg, "rnnt_max_symbols_per_step", rnnt_max_symbols))
    if rnnt_max_symbols <= 0:
        logger.warning(
            "rnnt_max_symbols_per_step=%d is not positive; defaulting to 32",
            rnnt_max_symbols,
        )
        rnnt_max_symbols = 32

    return DecodeOptions(
        strategy=strategy,
        beam_width=beam_width,
        beam_prune_threshold=prune_threshold,
        torchaudio_params=sanitized_torchaudio,
        use_duration_prior=use_duration_prior,
        rnnt_max_symbols_per_step=rnnt_max_symbols,
        rnnt_strategy=rnnt_strategy,
    )


def _build_torchaudio_decoder(
    options: DecodeOptions, logger: logging.Logger
) -> Optional[Any]:
    """Construct a torchaudio CTC decoder if dependencies are available."""

    try:
        from torchaudio.models.decoder import ctc_decoder
    except ModuleNotFoundError:
        logger.error("torchaudio is not installed; cannot use torchaudio decoding")
        return None
    except RuntimeError as exc:  # flashlight or other runtime dependency missing
        logger.error("torchaudio CTC decoder is unavailable: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.error("Failed to import torchaudio CTC decoder: %s", exc)
        return None

    beam_width = max(int(options.beam_width or 1), 1)

    params = dict(options.torchaudio_params)

    nbest = int(params.pop("nbest", 1) or 1)
    if nbest <= 0:
        logger.warning("Torchaudio nbest %d is not positive; defaulting to 1", nbest)
        nbest = 1

    beam_threshold = params.pop("beam_threshold", None)
    if beam_threshold is None:
        beam_threshold = options.beam_prune_threshold
    if beam_threshold is None:
        beam_threshold = 50.0
    else:
        beam_threshold = float(beam_threshold)

    decoder_kwargs: Dict[str, Any] = {
        "lexicon": params.pop("lexicon", None),
        "tokens": ["-"]
        + [BASE_VOCAB.get(index, "N") for index in range(1, max(BASE_VOCAB.keys(), default=0) + 1)],
        "nbest": nbest,
        "beam_size": beam_width,
        "beam_threshold": beam_threshold,
        "blank_token": params.pop("blank_token", "-"),
        "sil_token": params.pop("sil_token", "-"),
    }

    optional_float_keys = {"lm_weight", "word_score", "unk_score", "sil_score"}
    optional_int_keys = {"beam_size_token"}
    optional_bool_keys = {"log_add"}
    optional_passthrough = {"lm", "lm_dict", "unk_word"}

    for key in list(params.keys()):
        value = params.pop(key)
        if value is None:
            continue
        if key in optional_float_keys:
            decoder_kwargs[key] = float(value)
        elif key in optional_int_keys:
            decoder_kwargs[key] = int(value)
        elif key in optional_bool_keys:
            decoder_kwargs[key] = bool(value)
        elif key in optional_passthrough:
            decoder_kwargs[key] = value
        else:
            logger.warning("Ignoring unsupported torchaudio decoder option '%s'", key)

    try:
        return ctc_decoder(**decoder_kwargs)
    except Exception as exc:  # pragma: no cover - depends on external library
        logger.error("Failed to construct torchaudio CTC decoder: %s", exc)
        return None


def _torchaudio_decode(
    decoder: Any,
    log_probs: torch.Tensor,
    valid_length: int,
) -> str:
    """Decode a single example using a torchaudio decoder instance."""

    if valid_length <= 0:
        return ""

    trimmed = log_probs[:valid_length].to(torch.float32).detach().cpu().unsqueeze(0).contiguous()
    lengths = torch.tensor([valid_length], dtype=torch.int32)

    results = decoder(trimmed, lengths)
    if not results or not results[0]:
        return ""

    top_hypothesis = results[0][0]
    tokens = getattr(top_hypothesis, "tokens", None)
    if tokens is None or len(tokens) == 0:
        return ""

    sequence = []
    for token in tokens.tolist():
        sequence.append(BASE_VOCAB.get(int(token), "N"))
    return "".join(sequence)


def _resolve_precision(precision: str, device: torch.device) -> Tuple[torch.dtype, bool]:
    normalized = precision.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32, False
    if normalized in {"float16", "fp16"}:
        if device.type != "cuda":
            logging.getLogger("infer").warning(
                "Requested float16 precision but CUDA is unavailable; falling back to float32"
            )
            return torch.float32, False
        return torch.float16, True
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16, True
    raise ValueError("trainer.precision must be one of {'float32', 'float16', 'bfloat16'}")


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, non_blocking=True)
        elif isinstance(value, dict):
            result[key] = _move_to_device(value, device)
        elif isinstance(value, list):
            result[key] = [
                item.to(device=device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _infer_target_query_budget(batch: Dict[str, Any]) -> Optional[int]:
    move_tensor = batch.get("move")
    base_tensor = batch.get("base_index")
    if not (isinstance(move_tensor, torch.Tensor) and isinstance(base_tensor, torch.Tensor)):
        return None
    if move_tensor.shape != base_tensor.shape:
        return None

    target_mask = (move_tensor > 0) & (base_tensor > 0) & (base_tensor <= CTC_NUM_BASES)
    if target_mask.numel() == 0:
        return None

    counts = target_mask.sum(dim=-1)
    if counts.dim() >= 2:
        counts = counts.view(counts.size(0), -1).max(dim=1).values
    if counts.numel() == 0:
        return None

    max_required = int(counts.max().item())
    if max_required <= 0:
        return None
    return max_required


def _resolve_dataset(cfg: Any, split: str) -> ReadDataset:
    dataset_name = cfg.data.dataset.lower()
    if dataset_name not in {"", "read", "readdataset", "read_dataset"}:
        raise ValueError(f"Unsupported dataset type '{cfg.data.dataset}' for inference")

    params = cfg.data.params
    dataset_params = _dataset_kwargs(params)
    dataset_type = dataset_params.get("dataset_type", "legacy")

    if dataset_type in {"legacy", "npy"}:
        key = f"{split}_dir"
        path = _resolve_path(params.get(key))
        if not path:
            raise ValueError(
                f"Config is missing data.params.{key!s}; cannot construct '{split}' dataset for inference"
            )
        return ReadDataset(path, **dataset_params)

    dataset_root = _resolve_path(params.get("dataset_dir"))
    if dataset_root is None:
        raise ValueError("Monolithic datasets require 'dataset_dir' under data.params")

    explicit_segment = params.get("segment")
    if explicit_segment:
        return ReadDataset(dataset_root, segment_names=[explicit_segment], **dataset_params)

    split_config = params.get("split", {}) or {}
    split_seed = int(params.get("split_seed", cfg.seed))
    available_segments = list_block_segments(dataset_root)
    train_segments, val_segments, test_segments = _split_segments(
        available_segments, split_config, split_seed
    )

    selected: List[str]
    if split == "train":
        selected = train_segments
    elif split == "val":
        selected = val_segments
    else:
        selected = test_segments

    if not selected:
        raise ValueError(f"No segments available for split '{split}'")

    return ReadDataset(dataset_root, segment_names=selected, **dataset_params)


def _make_dataloader(cfg: Any, dataset: ReadDataset) -> DataLoader[Any]:
    batch_size = (
        cfg.data.test_batch_size
        or cfg.data.val_batch_size
        or cfg.data.batch_size
        or 1
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": cfg.data.num_workers,
        "shuffle": False,
        "collate_fn": collate_read_batch,
        "pin_memory": cfg.data.pin_memory,
    }
    if cfg.data.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.data.persistent_workers
        if cfg.data.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = cfg.data.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _resolve_checkpoint(cfg: Any, override: str) -> Optional[Path]:
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Checkpoint override '{override}' does not exist")

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    if checkpoint_dir is None:
        return None

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(cfg.output_dir) / checkpoint_path

    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("*.pt"))
        if candidates:
            return candidates[-1]
    return None


def _load_module_states(
    checkpoint: Optional[Dict[str, Any]],
    modules: Dict[str, torch.nn.Module],
    logger: logging.Logger,
) -> None:
    if checkpoint is None:
        logger.info("No checkpoint provided; running with randomly initialised weights")
        return

    for name, module in modules.items():
        state_dict = checkpoint.get(name)
        if state_dict is None:
            logger.warning("Checkpoint is missing '%s' parameters; leaving module unchanged", name)
            continue
        module.load_state_dict(state_dict, strict=False)
        logger.info("Loaded parameters for %s", name)


def _ctc_greedy_decode(indices: torch.Tensor, valid_length: int) -> str:
    sequence: List[str] = []
    prev = None
    for idx in indices[:valid_length].tolist():
        if idx == 0:
            prev = None
            continue
        if idx != prev:
            sequence.append(BASE_VOCAB.get(idx, "N"))
        prev = idx
    return "".join(sequence)


def _tensor_to_base_string(tensor: torch.Tensor) -> str:
    mapping = BASE_VOCAB
    bases: List[str] = []
    for value in tensor.tolist():
        base = mapping.get(int(value), "N")
        bases.append(base)
    return "".join(bases)


def _rnnt_tokens_to_string(tokens: Sequence[int]) -> str:
    return "".join(BASE_VOCAB.get(int(token), "N") for token in tokens if int(token) != 0)


def _rnnt_greedy_decode_batch(
    decoder: RNNTDecoder,
    outputs: Dict[str, Any],
    options: DecodeOptions,
    logger: logging.Logger,
) -> Tuple[List[str], List[int], List[int]]:
    hidden = outputs.get("embedding")
    if not isinstance(hidden, torch.Tensor):
        logger.warning("RNNT decoding requested but decoder outputs lacked 'embedding' tensor")
        return [], [], []

    logit_lengths = outputs.get("rnnt_logit_lengths")
    padding_mask = outputs.get("decoder_padding_mask")

    try:
        tokens, used_lengths = decoder.greedy_decode(
            hidden,
            input_lengths=logit_lengths,
            padding_mask=padding_mask,
            max_symbols_per_step=options.rnnt_max_symbols_per_step,
        )
    except Exception as exc:
        logger.error("Failed to run RNNT greedy decoding: %s", exc)
        return [], [], []

    token_sequences: Tuple[Tuple[int, ...], ...] = tokens if isinstance(tokens, tuple) else tuple(tokens)
    token_lengths = [len(seq) for seq in token_sequences]
    frame_lengths = used_lengths.detach().to("cpu").to(torch.int64).tolist()

    decoded_strings = [_rnnt_tokens_to_string(seq) for seq in token_sequences]
    return decoded_strings, frame_lengths, token_lengths


def _rnnt_beam_decode_batch(
    decoder: RNNTDecoder,
    outputs: Dict[str, Any],
    options: DecodeOptions,
    logger: logging.Logger,
) -> Tuple[List[str], List[int], List[int]]:
    hidden = outputs.get("embedding")
    if not isinstance(hidden, torch.Tensor):
        logger.warning("RNNT beam decoding requested but decoder outputs lacked 'embedding' tensor")
        return [], [], []

    logit_lengths = outputs.get("rnnt_logit_lengths")
    padding_mask = outputs.get("decoder_padding_mask")

    try:
        tokens, used_lengths = decoder.beam_decode(
            hidden,
            input_lengths=logit_lengths,
            padding_mask=padding_mask,
            beam_width=options.beam_width,
            max_symbols_per_step=options.rnnt_max_symbols_per_step,
        )
    except Exception as exc:
        logger.error("Failed to run RNNT beam decoding: %s", exc)
        return [], [], []

    token_sequences: Tuple[Tuple[int, ...], ...] = tokens if isinstance(tokens, tuple) else tuple(tokens)
    token_lengths = [len(seq) for seq in token_sequences]
    frame_lengths = used_lengths.detach().to("cpu").to(torch.int64).tolist()

    decoded_strings = [_rnnt_tokens_to_string(seq) for seq in token_sequences]
    return decoded_strings, frame_lengths, token_lengths


def _compute_alignment_stats(reference: str, hypothesis: str) -> AlignmentStats:
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if ref_len == 0 and hyp_len == 0:
        return AlignmentStats(
            distance=0,
            alignment_length=0,
            matches=0,
            reference_length=0,
            operations=[],
            reference_indices=[],
        )

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    back: List[List[str]] = [["" for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, ref_len + 1):
        ref_char = reference[i - 1]
        for j in range(1, hyp_len + 1):
            hyp_char = hypothesis[j - 1]
            substitution_cost = dp[i - 1][j - 1] + (0 if ref_char == hyp_char else 1)
            deletion_cost = dp[i - 1][j] + 1
            insertion_cost = dp[i][j - 1] + 1

            best_cost = substitution_cost
            op = "M" if ref_char == hyp_char else "S"
            if deletion_cost < best_cost:
                best_cost = deletion_cost
                op = "D"
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                op = "I"

            dp[i][j] = best_cost
            back[i][j] = op

    distance = dp[ref_len][hyp_len]
    matches = 0
    alignment_length = 0
    operations: List[str] = []
    reference_indices: List[Optional[int]] = []

    i, j = ref_len, hyp_len
    while i > 0 or j > 0:
        op = back[i][j]
        if not op:
            break
        operations.append(op)
        if op in {"M", "S"}:
            if op == "M":
                matches += 1
            reference_indices.append(i - 1)
            i -= 1
            j -= 1
        elif op == "D":
            reference_indices.append(i - 1)
            i -= 1
        elif op == "I":
            reference_indices.append(None)
            j -= 1
        else:
            break
        alignment_length += 1

    operations.reverse()
    reference_indices.reverse()

    return AlignmentStats(
        distance=distance,
        alignment_length=alignment_length,
        matches=matches,
        reference_length=ref_len,
        operations=operations,
        reference_indices=reference_indices,
    )


def _central_homopolymer_mask(sequence: str) -> List[bool]:
    length = len(sequence)
    if length == 0:
        return []

    center = (length - 1) // 2
    target_base = sequence[center]

    left = center
    while left > 0 and sequence[left - 1] == target_base:
        left -= 1

    right = center + 1
    while right < length and sequence[right] == target_base:
        right += 1

    mask = [False] * length
    for idx in range(left, right):
        mask[idx] = True
    return mask


def _homopolymer_mask(sequence: str, min_run: int, mode: str) -> List[bool]:
    if mode == "center":
        return _central_homopolymer_mask(sequence)

    if mode != "threshold":
        raise ValueError(f"Unsupported homopolymer mode: {mode}")

    if min_run <= 1:
        return [True] * len(sequence)

    mask = [False] * len(sequence)
    start = 0
    length = len(sequence)
    while start < length:
        end = start + 1
        current = sequence[start]
        while end < length and sequence[end] == current:
            end += 1
        run_length = end - start
        if run_length >= min_run:
            for idx in range(start, end):
                mask[idx] = True
        start = end
    return mask


def _compute_homopolymer_metrics(
    reference: str,
    alignment: AlignmentStats,
    min_run: int,
    mode: str,
) -> Optional[Dict[str, float]]:
    if alignment.reference_length == 0:
        return None

    mask = _homopolymer_mask(reference, min_run, mode)
    ref_hp_count = sum(1 for flag in mask if flag)
    if ref_hp_count == 0:
        return None

    matches = 0
    alignment_len = 0
    distance = 0
    hyp_len = 0
    last_ref_index: Optional[int] = None

    for op, ref_index in zip(alignment.operations, alignment.reference_indices):
        effective_index = ref_index if ref_index is not None else last_ref_index
        if ref_index is not None:
            last_ref_index = ref_index

        if effective_index is None or effective_index >= len(mask):
            continue
        if not mask[effective_index]:
            continue

        alignment_len += 1
        if op == "M":
            matches += 1
        else:
            distance += 1

        if op != "D":
            hyp_len += 1

    if alignment_len == 0:
        acc_aligned = 1.0
    else:
        acc_aligned = matches / float(alignment_len)

    acc_ref = matches / float(ref_hp_count) if ref_hp_count > 0 else 1.0
    norm_denom = max(ref_hp_count, hyp_len)
    normalized_distance = distance / float(norm_denom) if norm_denom > 0 else 0.0

    return {
        "matches": float(matches),
        "alignment_length": float(alignment_len),
        "ref_length": float(ref_hp_count),
        "hyp_length": float(hyp_len),
        "distance": float(distance),
        "acc_aligned": acc_aligned,
        "acc_ref": acc_ref,
        "normalized_distance": normalized_distance,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    configure_logging(cfg.output_dir)
    logger = logging.getLogger("infer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision, use_autocast = _resolve_precision(cfg.trainer.precision, device)

    logger.info("Device: %s | Precision: %s", device, precision)

    homopolymer_enabled = args.homopolymer_eval
    homopolymer_threshold = args.homopolymer_threshold
    homopolymer_mode = args.homopolymer_mode
    if homopolymer_mode != "center" and homopolymer_threshold < 1:
        logger.warning(
            "Homopolymer threshold %d is less than 1; defaulting to 1",
            homopolymer_threshold,
        )
        homopolymer_threshold = 1
    if homopolymer_enabled:
        if homopolymer_mode == "center":
            logger.info(
                "Homopolymer evaluation enabled | mode=center (midpoint reference run)",
            )
        else:
            logger.info(
                "Homopolymer evaluation enabled | mode=threshold | min_run=%d",
                homopolymer_threshold,
            )

    torchaudio_overrides: Dict[str, Any] = {}
    if args.torchaudio_nbest is not None:
        torchaudio_overrides["nbest"] = args.torchaudio_nbest
    if args.torchaudio_beam_threshold is not None:
        torchaudio_overrides["beam_threshold"] = args.torchaudio_beam_threshold

    decode_options = _resolve_decode_options(
        cfg,
        logger,
        strategy_override=args.decode_strategy,
        beam_width_override=args.beam_width,
        prune_override=args.beam_prune_threshold,
        torchaudio_overrides=torchaudio_overrides or None,
    )
    torchaudio_decoder: Optional[Any] = None
    if decode_options.rnnt_strategy == "beam":
        logger.info(
            "RNNT decode strategy: beam | beam_width=%d | max_symbols_per_step=%d",
            decode_options.beam_width,
            decode_options.rnnt_max_symbols_per_step,
        )
    if decode_options.strategy == "beam":
        logger.info(
            "Decode strategy: beam | beam_width=%d | prune_threshold=%s",
            decode_options.beam_width,
            "none"
            if decode_options.beam_prune_threshold is None
            else f"{decode_options.beam_prune_threshold:.3f}",
        )
    elif decode_options.strategy == "torchaudio":
        torchaudio_decoder = _build_torchaudio_decoder(decode_options, logger)
        if torchaudio_decoder is None:
            logger.warning("Falling back to greedy decoding due to unavailable torchaudio decoder")
            decode_options = DecodeOptions(
                strategy="greedy",
                beam_width=decode_options.beam_width,
                beam_prune_threshold=decode_options.beam_prune_threshold,
            )
            logger.info("Decode strategy: greedy")
        else:
            nbest = decode_options.torchaudio_params.get("nbest", 1)
            beam_threshold = decode_options.torchaudio_params.get("beam_threshold")
            if beam_threshold is None:
                beam_threshold = (
                    decode_options.beam_prune_threshold
                    if decode_options.beam_prune_threshold is not None
                    else 50.0
                )
            logger.info(
                "Decode strategy: torchaudio | beam_width=%d | nbest=%d | beam_threshold=%s",
                decode_options.beam_width,
                nbest,
                "none" if beam_threshold is None else f"{beam_threshold:.3f}",
            )
    elif decode_options.strategy == "viterbi":
        logger.info("Decode strategy: viterbi (decoder-supplied Viterbi alignments)")
    else:
        logger.info("Decode strategy: greedy")

    encoder = ENCODER_REGISTRY.create(cfg.encoder.name, **_component_kwargs(cfg.encoder)).to(device)
    fusion = FUSION_REGISTRY.create(cfg.fusion.name, **_component_kwargs(cfg.fusion)).to(device)
    aggregator = AGGREGATOR_REGISTRY.create(
        cfg.aggregator.name, **_component_kwargs(cfg.aggregator)
    ).to(device)
    decoder = DECODER_REGISTRY.create(
        cfg.decoder.name, **_component_kwargs(cfg.decoder)
    ).to(device)
    if decoder.__class__.__name__ == "CTCCRFDecoder" and not getattr(decoder, "return_viterbi", False):
        logger.info("Enabling Viterbi outputs for CTC-CRF decoder during inference")
        decoder.return_viterbi = True

    for module in (encoder, fusion, aggregator, decoder):
        module.eval()

    checkpoint_path = _resolve_checkpoint(cfg, args.ckpt)
    checkpoint_state: Optional[Dict[str, Any]] = None
    if checkpoint_path is not None:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint_state = load_checkpoint(str(checkpoint_path), map_location=device)
    _load_module_states(
        checkpoint_state,
        {
            "encoder": encoder,
            "fusion": fusion,
            "aggregator": aggregator,
            "decoder": decoder,
        },
        logger,
    )

    dataset = _resolve_dataset(cfg, args.split)
    dataloader = _make_dataloader(cfg, dataset)
    logger.info(
        "Running inference on %s split at %s (%d segments)",
        args.split,
        dataset.segment_path,
        len(dataset),
    )

    if use_autocast:
        def autocast_cm():
            return torch.autocast(device_type=device.type, dtype=precision)

    else:
        def autocast_cm():
            return nullcontext()

    decoded_records: List[DecodedRecord] = []
    acc_aligned_list: List[float] = []
    acc_ref_list: List[float] = []
    distance_list: List[float] = []
    normalized_distance_list: List[float] = []
    total_matches = 0.0
    total_alignment_length = 0.0
    total_ref_length = 0.0
    samples_with_reference = 0
    total_logit_lengths: List[int] = []
    total_non_blank: List[int] = []
    total_char_lengths: List[int] = []
    total_frame_lengths: List[int] = []
    collapsed_lengths: List[int] = []
    hp_acc_aligned_list: List[float] = []
    hp_acc_ref_list: List[float] = []
    hp_distance_list: List[float] = []
    hp_normalized_distance_list: List[float] = []
    hp_total_matches = 0.0
    hp_total_alignment_length = 0.0
    hp_total_ref_length = 0.0
    hp_samples_with_reference = 0

    progress = tqdm(dataloader, desc=f"Inference ({args.split})", unit="batch")
    with torch.no_grad():
        for batch in progress:
            raw_segment_ids = batch.get("segment_id", [])
            if isinstance(raw_segment_ids, list):
                segment_ids = raw_segment_ids
            elif isinstance(raw_segment_ids, str):
                segment_ids = [raw_segment_ids]
            else:
                segment_ids = list(raw_segment_ids)
            reference_strings = batch.get("reference_string", [])
            moved = _move_to_device(batch, device)

            reference_targets, reference_lengths = _gather_reference(moved)
            reference_lengths_cpu = reference_lengths.detach().to("cpu")
            reference_targets_cpu = [tensor.detach().to("cpu") for tensor in reference_targets]

            target_query_budget = _infer_target_query_budget(moved)
            with autocast_cm():
                encoded = encoder(moved)
                if target_query_budget is not None:
                    base_queries = int(getattr(fusion, "num_queries", target_query_budget))
                    encoded["target_query_count"] = max(base_queries, target_query_budget)
                fused = fusion(encoded)
                aggregated = aggregator(fused)
                if "decoder_padding_mask" not in aggregated:
                    pad_mask = moved.get("decoder_padding_mask")
                    if pad_mask is not None:
                        aggregated["decoder_padding_mask"] = pad_mask
                outputs = decoder(aggregated)

            rnnt_logits = outputs.get("rnnt_logits")
            logits = outputs.get("ctc_logits")
            lengths = outputs.get("ctc_logit_lengths")
            gather_indices = outputs.get("ctc_logit_gather_indices")
            padding_mask = outputs.get("decoder_padding_mask")
            handled_batch = False
            if rnnt_logits is not None and isinstance(decoder, RNNTDecoder):
                handled_batch = True
                decode_fn = (
                    _rnnt_beam_decode_batch
                    if decode_options.rnnt_strategy == "beam"
                    else _rnnt_greedy_decode_batch
                )
                decoded_batch, frame_lengths, char_lengths_batch = decode_fn(
                    decoder, outputs, decode_options, logger
                )
                for idx, seg_id in enumerate(segment_ids):
                    if idx >= len(decoded_batch) or idx >= len(frame_lengths):
                        break
                    decoded = decoded_batch[idx]
                    logit_len = int(frame_lengths[idx]) if idx < len(frame_lengths) else 0
                    char_len = (
                        int(char_lengths_batch[idx])
                        if idx < len(char_lengths_batch)
                        else len(decoded)
                    )
                    total_logit_lengths.append(logit_len)
                    total_non_blank.append(char_len)
                    total_char_lengths.append(char_len)
                    total_frame_lengths.append(logit_len)
                    decoded_records.append(
                        DecodedRecord(
                            segment=seg_id,
                            sequence=decoded,
                            logit_length=logit_len,
                            non_blank=char_len,
                            run_total=char_len,
                            frame_run_total=logit_len,
                        )
                    )

                    if idx < len(reference_targets_cpu):
                        target_len = (
                            int(reference_lengths_cpu[idx].item())
                            if idx < reference_lengths_cpu.numel()
                            else reference_targets_cpu[idx].numel()
                        )
                        if target_len <= 0:
                            raise ValueError(
                                f"Reference sequence for segment '{seg_id}' has non-positive length ({target_len})"
                            )

                        reference_sequence = _tensor_to_base_string(
                            reference_targets_cpu[idx][:target_len]
                        )
                        reference_text = ""
                        if isinstance(reference_strings, list) and idx < len(reference_strings):
                            reference_text = str(reference_strings[idx])
                        if reference_text:
                            if reference_text != reference_sequence:
                                logger.warning(
                                    "Reference text mismatch for segment '%s'; using FASTA string",
                                    seg_id,
                                )
                            reference_sequence = reference_text

                        alignment_stats = _compute_alignment_stats(
                            reference_sequence,
                            decoded,
                        )
                        distance = alignment_stats.distance
                        alignment_len = alignment_stats.alignment_length
                        matches = alignment_stats.matches
                        ref_len = alignment_stats.reference_length

                        if alignment_len == 0:
                            acc_aligned = 1.0
                        else:
                            acc_aligned = matches / float(alignment_len)
                        acc_ref = matches / float(ref_len) if ref_len > 0 else 1.0
                        norm_denom = max(ref_len, len(decoded))
                        normalized_distance = distance / float(norm_denom) if norm_denom > 0 else 0.0

                        acc_aligned_list.append(acc_aligned)
                        acc_ref_list.append(acc_ref)
                        distance_list.append(float(distance))
                        normalized_distance_list.append(normalized_distance)

                        total_matches += matches
                        total_alignment_length += alignment_len
                        total_ref_length += ref_len
                        samples_with_reference += 1

                        if homopolymer_enabled:
                            hp_metrics = _compute_homopolymer_metrics(
                                reference_sequence,
                                alignment_stats,
                                homopolymer_threshold,
                                homopolymer_mode,
                            )
                            if hp_metrics is not None:
                                hp_acc_aligned_list.append(hp_metrics["acc_aligned"])
                                hp_acc_ref_list.append(hp_metrics["acc_ref"])
                                hp_distance_list.append(hp_metrics["distance"])
                                hp_normalized_distance_list.append(
                                    hp_metrics["normalized_distance"]
                                )
                                hp_total_matches += hp_metrics["matches"]
                                hp_total_alignment_length += hp_metrics["alignment_length"]
                                hp_total_ref_length += hp_metrics["ref_length"]
                                hp_samples_with_reference += 1

            if (
                not handled_batch
                and logits is not None
                and lengths is not None
                and decode_options.strategy != "viterbi"
            ):
                handled_batch = True
                valid_logits = _extract_valid_ctc_logits(logits, lengths, gather_indices, padding_mask)
                duration_priors_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                if (
                    decode_options.strategy == "beam"
                    and decode_options.use_duration_prior
                ):
                    duration_logits = outputs.get("duration_logits")
                    if isinstance(duration_logits, torch.Tensor):
                        duration_compact = _extract_valid_duration_logits(
                            duration_logits, lengths, gather_indices, padding_mask
                        )
                        stay_log = torch.logsigmoid(-duration_compact.to(torch.float32))
                        advance_log = torch.logsigmoid(duration_compact.to(torch.float32))
                        duration_priors_batch = (stay_log, advance_log)
                    else:
                        logger.debug(
                            "Duration prior requested but decoder outputs lacked 'duration_logits'"
                        )
                best_indices = valid_logits.detach().to(torch.float32).argmax(dim=-1).to("cpu")
                length_tensor = lengths.detach().to("cpu")
                for idx, seg_id in enumerate(segment_ids):
                    if idx >= best_indices.size(0):
                        break
                    valid = int(length_tensor[idx].item()) if idx < length_tensor.numel() else best_indices.size(1)
                    total_logit_lengths.append(valid)
                    non_blank = sum(1 for token in best_indices[idx][:valid].tolist() if token != 0)
                    total_non_blank.append(non_blank)
                    if decode_options.strategy == "beam" and valid > 0:
                        log_prob_slice = torch.log_softmax(
                            valid_logits[idx, :valid].to(torch.float32), dim=-1
                        )
                        sample_priors = None
                        if duration_priors_batch is not None:
                            stay_batch, advance_batch = duration_priors_batch
                            sample_priors = (stay_batch[idx], advance_batch[idx])
                        decoded = _ctc_prefix_beam_search(
                            log_prob_slice,
                            valid_length=valid,
                            beam_width=decode_options.beam_width,
                            prune_threshold=decode_options.beam_prune_threshold,
                            duration_priors=sample_priors,
                        )
                    elif decode_options.strategy == "torchaudio" and torchaudio_decoder is not None:
                        log_prob_slice = torch.log_softmax(
                            valid_logits[idx, :valid].to(torch.float32), dim=-1
                        )
                        decoded = _torchaudio_decode(torchaudio_decoder, log_prob_slice, valid)
                    elif decode_options.strategy in {"beam", "torchaudio"}:
                        decoded = ""
                    else:
                        decoded = _ctc_greedy_decode(best_indices[idx], valid)
                    decoded_records.append(
                        DecodedRecord(
                            segment=seg_id,
                            sequence=decoded,
                            logit_length=valid,
                            non_blank=non_blank,
                            run_total=non_blank,
                            frame_run_total=valid,
                        )
                    )
                    total_char_lengths.append(non_blank)
                    total_frame_lengths.append(valid)

                    if idx < len(reference_targets_cpu):
                        target_len = (
                            int(reference_lengths_cpu[idx].item())
                            if idx < reference_lengths_cpu.numel()
                            else reference_targets_cpu[idx].numel()
                        )
                        if target_len <= 0:
                            raise ValueError(
                                f"Reference sequence for segment '{seg_id}' has non-positive length ({target_len})"
                            )

                        reference_sequence = _tensor_to_base_string(
                            reference_targets_cpu[idx][:target_len]
                        )
                        reference_text = ""
                        if isinstance(reference_strings, list) and idx < len(reference_strings):
                            reference_text = str(reference_strings[idx])
                        if reference_text:
                            if reference_text != reference_sequence:
                                logger.warning(
                                    "Reference text mismatch for segment '%s'; using FASTA string",
                                    seg_id,
                                )
                            reference_sequence = reference_text

                        alignment_stats = _compute_alignment_stats(
                            reference_sequence,
                            decoded,
                        )
                        distance = alignment_stats.distance
                        alignment_len = alignment_stats.alignment_length
                        matches = alignment_stats.matches
                        ref_len = alignment_stats.reference_length

                        if alignment_len == 0:
                            acc_aligned = 1.0
                        else:
                            acc_aligned = matches / float(alignment_len)
                        acc_ref = matches / float(ref_len) if ref_len > 0 else 1.0
                        norm_denom = max(ref_len, len(decoded))
                        normalized_distance = distance / float(norm_denom) if norm_denom > 0 else 0.0

                        acc_aligned_list.append(acc_aligned)
                        acc_ref_list.append(acc_ref)
                        distance_list.append(float(distance))
                        normalized_distance_list.append(normalized_distance)

                        total_matches += matches
                        total_alignment_length += alignment_len
                        total_ref_length += ref_len
                        samples_with_reference += 1

                        if homopolymer_enabled:
                            hp_metrics = _compute_homopolymer_metrics(
                                reference_sequence,
                                alignment_stats,
                                homopolymer_threshold,
                                homopolymer_mode,
                            )
                            if hp_metrics is not None:
                                hp_acc_aligned_list.append(hp_metrics["acc_aligned"])
                                hp_acc_ref_list.append(hp_metrics["acc_ref"])
                                hp_distance_list.append(hp_metrics["distance"])
                                hp_normalized_distance_list.append(
                                    hp_metrics["normalized_distance"]
                                )
                                hp_total_matches += hp_metrics["matches"]
                                hp_total_alignment_length += hp_metrics["alignment_length"]
                                hp_total_ref_length += hp_metrics["ref_length"]
                                hp_samples_with_reference += 1

            sequences = outputs.get("viterbi_sequence")
            if not handled_batch and sequences is not None:
                handled_batch = True
                char_run_lengths_out = outputs.get("viterbi_char_run_length") or []
                frame_run_lengths_out = outputs.get("viterbi_frame_run_length") or []
                collapsed_tokens_out = outputs.get("viterbi_collapsed_tokens") or []
                emissions = outputs.get("ctc_emissions")
                if decode_options.strategy not in {"greedy", "viterbi"}:
                    logger.warning(
                        "Decode strategy '%s' is not supported for CTC-CRF outputs; using Viterbi results",
                        decode_options.strategy,
                    )

                mask_lengths: Optional[List[int]] = None
                if padding_mask is not None:
                    target_steps = (
                        emissions.size(1)
                        if emissions is not None and emissions.dim() >= 2
                        else padding_mask.size(-1)
                    )
                    compact_mask = _compact_padding_mask(padding_mask, target_steps)
                    valid_mask = torch.logical_not(compact_mask)
                    mask_lengths = valid_mask.sum(dim=1).to("cpu").tolist()
                elif emissions is not None:
                    mask_lengths = [emissions.size(1)] * emissions.size(0)

                for idx, seg_id in enumerate(segment_ids):
                    if idx >= len(sequences):
                        break
                    seq_tokens = sequences[idx] if sequences[idx] is not None else []
                    char_runs = _select_sequence_item(char_run_lengths_out, idx)
                    frame_runs = _select_sequence_item(frame_run_lengths_out, idx)
                    collapsed_tokens_seq = _select_sequence_item(collapsed_tokens_out, idx)
                    decode_result = _decode_viterbi_sequence(
                        seq_tokens,
                        char_run_lengths=char_runs,
                        frame_run_lengths=frame_runs,
                        collapsed_tokens=collapsed_tokens_seq,
                    )
                    decoded = decode_result.collapsed
                    stats = _log_viterbi_stats(logger, seg_id, decode_result)
                    collapsed_len = decode_result.collapsed_length
                    expanded_sequence = decode_result.expanded
                    expanded_length = decode_result.expanded_length
                    char_total = int(stats["char_total"])
                    frame_total = int(stats["frame_total"])
                    run_total = char_total
                    logit_len = frame_total
                    if mask_lengths is not None and idx < len(mask_lengths):
                        logit_len = max(logit_len, int(mask_lengths[idx]))
                    if logit_len <= 0:
                        logit_len = max(frame_total, char_total, collapsed_len)
                    non_blank = char_total
                    total_logit_lengths.append(logit_len)
                    total_non_blank.append(non_blank)
                    total_char_lengths.append(char_total)
                    total_frame_lengths.append(frame_total)
                    collapsed_lengths.append(collapsed_len)
                    decoded_records.append(
                        DecodedRecord(
                            segment=seg_id,
                            sequence=decoded,
                            logit_length=logit_len,
                            non_blank=non_blank,
                            run_total=run_total,
                            frame_run_total=frame_total,
                            same_base_reenter_count=int(stats["same_base_reenter"]),
                            collapsed_sequence=decode_result.collapsed,
                            collapsed_length=collapsed_len,
                            expanded_sequence=expanded_sequence,
                            expanded_length=expanded_length,
                        )
                    )

                    if idx < len(reference_targets_cpu):
                        target_len = (
                            int(reference_lengths_cpu[idx].item())
                            if idx < reference_lengths_cpu.numel()
                            else reference_targets_cpu[idx].numel()
                        )
                        if target_len <= 0:
                            raise ValueError(
                                f"Reference sequence for segment '{seg_id}' has non-positive length ({target_len})"
                            )

                        reference_sequence = _tensor_to_base_string(
                            reference_targets_cpu[idx][:target_len]
                        )
                        reference_text = ""
                        if isinstance(reference_strings, list) and idx < len(reference_strings):
                            reference_text = str(reference_strings[idx])
                        if reference_text:
                            if reference_text != reference_sequence:
                                logger.warning(
                                    "Reference text mismatch for segment '%s'; using FASTA string",
                                    seg_id,
                                )
                            reference_sequence = reference_text

                        alignment_stats = _compute_alignment_stats(
                            reference_sequence,
                            decoded,
                        )
                        distance = alignment_stats.distance
                        alignment_len = alignment_stats.alignment_length
                        matches = alignment_stats.matches
                        ref_len = alignment_stats.reference_length

                        if alignment_len == 0:
                            acc_aligned = 1.0
                        else:
                            acc_aligned = matches / float(alignment_len)
                        acc_ref = matches / float(ref_len) if ref_len > 0 else 1.0
                        norm_denom = max(ref_len, len(decoded))
                        normalized_distance = distance / float(norm_denom) if norm_denom > 0 else 0.0

                        acc_aligned_list.append(acc_aligned)
                        acc_ref_list.append(acc_ref)
                        distance_list.append(float(distance))
                        normalized_distance_list.append(normalized_distance)

                        total_matches += matches
                        total_alignment_length += alignment_len
                        total_ref_length += ref_len
                        samples_with_reference += 1

                        if homopolymer_enabled:
                            hp_metrics = _compute_homopolymer_metrics(
                                reference_sequence,
                                alignment_stats,
                                homopolymer_threshold,
                                homopolymer_mode,
                            )
                            if hp_metrics is not None:
                                hp_acc_aligned_list.append(hp_metrics["acc_aligned"])
                                hp_acc_ref_list.append(hp_metrics["acc_ref"])
                                hp_distance_list.append(hp_metrics["distance"])
                                hp_normalized_distance_list.append(
                                    hp_metrics["normalized_distance"]
                                )
                                hp_total_matches += hp_metrics["matches"]
                                hp_total_alignment_length += hp_metrics["alignment_length"]
                                hp_total_ref_length += hp_metrics["ref_length"]
                                hp_samples_with_reference += 1

            if not handled_batch:
                logger.warning("Decoder outputs do not include CTC logits or Viterbi sequences; skipping batch")

    logger.info("Inference finished. Decoded %d segments", len(decoded_records))
    for record in decoded_records[: min(10, len(decoded_records))]:
        collapsed_msg = ""
        if record.collapsed_sequence is not None:
            collapsed_msg = (
                " | collapsed_len=%d | collapsed_sequence=%s"
                % (
                    record.collapsed_length
                    if record.collapsed_length is not None
                    else len(record.collapsed_sequence),
                    record.collapsed_sequence,
                )
            )
        expanded_msg = ""
        if record.expanded_sequence is not None:
            expanded_len = (
                record.expanded_length
                if record.expanded_length is not None
                else len(record.expanded_sequence)
            )
            expanded_msg = " | expanded_len=%d" % expanded_len
        frame_msg = ""
        if record.frame_run_total is not None:
            frame_msg = f" | frame_total={record.frame_run_total}"
        reenter_msg = ""
        if record.same_base_reenter_count is not None:
            reenter_msg = f" | same_base_reenter={record.same_base_reenter_count}"
        logger.info(
            "Segment %s | decoded_length=%d | logit_length=%d | non_blank=%d | run_total=%d%s%s%s%s",
            record.segment,
            len(record.sequence),
            record.logit_length,
            record.non_blank,
            record.run_total,
            frame_msg,
            reenter_msg,
            collapsed_msg,
            expanded_msg,
        )

    if decoded_records:
        total_segments = len(decoded_records)
        zero_length = sum(1 for record in decoded_records if len(record.sequence) == 0)
        mean_collapsed = (
            sum(len(record.sequence) for record in decoded_records) / total_segments
        )
        mean_logit = sum(total_logit_lengths) / total_segments if total_logit_lengths else 0.0
        mean_non_blank = sum(total_non_blank) / total_segments if total_non_blank else 0.0
        mean_char_total = sum(total_char_lengths) / total_segments if total_char_lengths else 0.0
        mean_frame_total = sum(total_frame_lengths) / total_segments if total_frame_lengths else 0.0
        mean_tracked_collapsed = (
            sum(collapsed_lengths) / len(collapsed_lengths)
            if collapsed_lengths
            else 0.0
        )
        logger.info(
            "Decoded summary | segments=%d | mean_collapsed_len=%.2f | mean_char_total=%.2f | "
            "mean_frame_total=%.2f | mean_logit_len=%.2f | mean_non_blank=%.2f | "
            "tracked_collapsed_len=%.2f | zero_length=%d (%.2f%%)",
            total_segments,
            mean_collapsed,
            mean_char_total,
            mean_frame_total,
            mean_logit,
            mean_non_blank,
            mean_tracked_collapsed,
            zero_length,
            (zero_length / total_segments) * 100.0,
        )

    if samples_with_reference > 0:
        mean_acc_aligned = sum(acc_aligned_list) / samples_with_reference
        mean_acc_ref = sum(acc_ref_list) / samples_with_reference
        mean_distance = sum(distance_list) / samples_with_reference
        mean_normalized_distance = (
            sum(normalized_distance_list) / samples_with_reference
            if normalized_distance_list
            else 0.0
        )

        overall_acc_aligned = (
            total_matches / total_alignment_length if total_alignment_length > 0 else 1.0
        )
        overall_acc_ref = total_matches / total_ref_length if total_ref_length > 0 else 1.0

        logger.info(
            "Evaluation metrics computed on %d segments with ground truth:", samples_with_reference
        )
        logger.info("  Mean Acc@aligned: %.4f", mean_acc_aligned)
        logger.info("  Overall base-level accuracy (alignment-aware): %.4f", overall_acc_aligned)
        logger.info("  Mean Acc@pos (reference): %.4f", mean_acc_ref)
        logger.info("  Overall Acc@pos (reference): %.4f", overall_acc_ref)
        logger.info("  Mean Levenshtein distance: %.4f", mean_distance)
        logger.info(
            "  Mean normalized Levenshtein distance: %.4f", mean_normalized_distance
        )
    else:
        logger.info("No reference sequences with positive length were available; skipping metrics")

    if homopolymer_enabled:
        if hp_samples_with_reference > 0:
            hp_mean_acc_aligned = sum(hp_acc_aligned_list) / hp_samples_with_reference
            hp_mean_acc_ref = sum(hp_acc_ref_list) / hp_samples_with_reference
            hp_mean_distance = sum(hp_distance_list) / hp_samples_with_reference
            hp_mean_normalized_distance = (
                sum(hp_normalized_distance_list) / hp_samples_with_reference
                if hp_normalized_distance_list
                else 0.0
            )

            hp_overall_acc_aligned = (
                hp_total_matches / hp_total_alignment_length
                if hp_total_alignment_length > 0
                else 1.0
            )
            hp_overall_acc_ref = (
                hp_total_matches / hp_total_ref_length
                if hp_total_ref_length > 0
                else 1.0
            )

            if homopolymer_mode == "center":
                logger.info(
                    "Homopolymer metrics computed on %d segment(s) (mode=center)",
                    hp_samples_with_reference,
                )
            else:
                logger.info(
                    "Homopolymer metrics computed on %d segment(s) (min_run=%d):",
                    hp_samples_with_reference,
                    homopolymer_threshold,
                )
            logger.info("  [Homopolymer] Mean Acc@aligned: %.4f", hp_mean_acc_aligned)
            logger.info(
                "  [Homopolymer] Overall base-level accuracy (alignment-aware): %.4f",
                hp_overall_acc_aligned,
            )
            logger.info("  [Homopolymer] Mean Acc@pos (reference): %.4f", hp_mean_acc_ref)
            logger.info(
                "  [Homopolymer] Overall Acc@pos (reference): %.4f", hp_overall_acc_ref
            )
            logger.info("  [Homopolymer] Mean Levenshtein distance: %.4f", hp_mean_distance)
            logger.info(
                "  [Homopolymer] Mean normalized Levenshtein distance: %.4f",
                hp_mean_normalized_distance,
            )
        else:
            if homopolymer_mode == "center":
                logger.info(
                    "Homopolymer evaluation enabled but no midpoint homopolymer regions were found; skipping homopolymer metrics",
                )
            else:
                logger.info(
                    "Homopolymer evaluation enabled but no regions with length >= %d were found; skipping homopolymer metrics",
                    homopolymer_threshold,
                )


if __name__ == "__main__":
    main()