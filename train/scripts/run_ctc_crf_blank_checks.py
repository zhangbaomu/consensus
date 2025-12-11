#!/usr/bin/env python3
"""Utility script to sanity-check the blank CTC-CRF pipeline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import ExperimentConfig, load_config
from ngpathfinder.data import ReadDataset, collate_read_batch
from ngpathfinder.losses import LOSS_REGISTRY
from ngpathfinder.modules.aggregator import AGGREGATOR_REGISTRY
from ngpathfinder.modules.decoder import CTCCRFDecoder, DECODER_REGISTRY
from ngpathfinder.modules.encoder import ENCODER_REGISTRY
from ngpathfinder.modules.fusion import FUSION_REGISTRY

TensorDict = MutableMapping[str, Any]


def _infer_target_query_budget(batch: TensorDict) -> Optional[int]:
    move_tensor = batch.get("move")
    base_tensor = batch.get("base_index")
    if not (isinstance(move_tensor, torch.Tensor) and isinstance(base_tensor, torch.Tensor)):
        return None
    if move_tensor.shape != base_tensor.shape:
        return None

    target_mask = (move_tensor > 0) & (base_tensor > 0) & (base_tensor <= 4)
    if target_mask.numel() == 0:
        return None

    counts = target_mask.sum(dim=-1)
    if counts.dim() >= 2:
        counts = counts.view(counts.size(0), -1).max(dim=1).values
    if counts.numel() == 0:
        return None

    max_required = int(counts.max().item())
    return max_required if max_required > 0 else None


def _clone_batch(batch: TensorDict) -> TensorDict:
    result: TensorDict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        elif isinstance(value, dict):
            result[key] = _clone_batch(value)  # type: ignore[arg-type]
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _move_to_device(batch: TensorDict, device: torch.device) -> TensorDict:
    result: TensorDict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device)
        elif isinstance(value, list):
            converted: List[Any] = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    converted.append(item.to(device=device))
                else:
                    converted.append(item)
            result[key] = converted
        elif isinstance(value, dict):
            result[key] = _move_to_device(value, device)
        else:
            result[key] = value
    return result


def _build_dataset(cfg: ExperimentConfig, segment_index: int) -> Tuple[TensorDict, torch.Tensor]:
    params = cfg.data.params
    train_dir = params.get("train_dir")
    if not train_dir:
        raise ValueError("data.params.train_dir must be specified for the sanity checks")
    train_path = (REPO_ROOT / train_dir).resolve()

    dataset_kwargs = {
        "max_mv_len": params.get("max_mv_len"),
        "max_reads_per_segment": params.get("max_reads_per_segment"),
        "use_fasta_reference": params.get("use_fasta_reference", True),
        "fasta_glob_patterns": params.get("fasta_glob_patterns"),
        "ambiguous_base_policy": params.get("ambiguous_base_policy", "error"),
        "fasta_sequence_policy": params.get("fasta_sequence_policy", "first"),
        "suppress_mv_len_warnings": params.get("suppress_mv_len_warnings", False),
        "use_fastq_base_sequence": params.get("use_fastq_base_sequence", True),
        "use_read_flag": params.get("use_read_flag", True),
    }

    dataset = ReadDataset(train_path, **dataset_kwargs)
    if segment_index < 0 or segment_index >= len(dataset):
        raise IndexError(
            f"Segment index {segment_index} is out of bounds for dataset of size {len(dataset)}"
        )
    sample = dataset[segment_index]
    batch = collate_read_batch([sample])
    reference_lengths = batch["reference_lengths"].clone()
    return batch, reference_lengths


def _instantiate_modules(cfg: ExperimentConfig) -> Tuple[torch.nn.Module, ...]:
    encoder = ENCODER_REGISTRY.create(cfg.encoder.name, **cfg.encoder.params)
    fusion = FUSION_REGISTRY.create(cfg.fusion.name, **cfg.fusion.params)
    aggregator = AGGREGATOR_REGISTRY.create(cfg.aggregator.name, **cfg.aggregator.params)
    decoder = DECODER_REGISTRY.create(cfg.decoder.name, **cfg.decoder.params)
    return encoder, fusion, aggregator, decoder


def _run_modules(
    modules: Tuple[torch.nn.Module, ...],
    batch: TensorDict,
    target_query_budget: Optional[int],
) -> TensorDict:
    encoder, fusion, aggregator, decoder = modules

    encoded = encoder(batch)
    if target_query_budget is not None:
        base_queries = int(getattr(fusion, "num_queries", target_query_budget))
        encoded["target_query_count"] = max(base_queries, target_query_budget)
    fused = fusion(encoded)
    aggregated = aggregator(fused)
    if "decoder_padding_mask" not in aggregated and "decoder_padding_mask" in batch:
        aggregated["decoder_padding_mask"] = batch["decoder_padding_mask"]
    outputs = decoder(aggregated)
    return outputs


def _summarise_viterbi(
    run_bases: Sequence[Sequence[int]],
    collapsed_tokens: Sequence[Sequence[int]],
    frame_rle: Sequence[Sequence[Tuple[str, int]]],
    reference_lengths: Sequence[int],
) -> List[Dict[str, float]]:
    # When the decoder does not provide the run-level bases, fall back to the
    # collapsed tokens so the diagnostic still reports a useful length.
    if not run_bases:
        run_bases = collapsed_tokens
    summary: List[Dict[str, float]] = []
    for idx, (bases, collapsed, runs, ref_len) in enumerate(
        zip(run_bases, collapsed_tokens, frame_rle, reference_lengths)
    ):
        total_frames = sum(length for _, length in runs)
        blank_frames = sum(length for char, length in runs if char == "-")
        blank_rate = (blank_frames / total_frames) if total_frames else float("nan")
        ref = float(ref_len) if ref_len else float("nan")
        collapsed_len = float(len(bases))
        expanded_len = float(len(collapsed))
        ratio = (collapsed_len / ref) if ref and not math.isnan(ref) else float("nan")
        summary.append(
            {
                "sample": float(idx),
                "blank_rate": blank_rate,
                "collapsed_len": collapsed_len,
                "expanded_len": expanded_len,
                "ref_len": ref,
                "collapsed_ref_ratio": ratio,
            }
        )
    return summary


def _print_viterbi_summary(title: str, summary: List[Dict[str, float]]) -> None:
    print(title)
    for item in summary:
        sample = int(item["sample"])
        blank_rate = item["blank_rate"]
        ratio = item["collapsed_ref_ratio"]
        collapsed_len = item["collapsed_len"]
        expanded_len = item.get("expanded_len", float("nan"))
        ref_len = item["ref_len"]
        print(
            f"  sample={sample:02d} blank_rate={blank_rate:.3f} collapsed_len={collapsed_len:.1f} "
            f"expanded_len={expanded_len:.1f} "
            f"ref_len={ref_len:.1f} ratio={ratio:.3f}"
        )


def check_log_probabilities(
    modules: Tuple[torch.nn.Module, ...],
    base_batch: TensorDict,
    target_query_budget: Optional[int],
    device: torch.device,
) -> None:
    modules_eval = tuple(module.eval() for module in modules)
    batch = _move_to_device(_clone_batch(base_batch), device)
    with torch.no_grad():
        outputs = _run_modules(modules_eval, batch, target_query_budget)
    logits = outputs.get("ctc_blank_logits")
    if logits is None:
        raise KeyError("Decoder outputs did not include 'ctc_blank_logits'")
    logsumexp = torch.logsumexp(logits, dim=-1)
    print("CHECK 1: log-sum-exp of blank logits (ideal ~0.0)")
    print(
        f"  shape={tuple(logsumexp.shape)} min={logsumexp.min().item():.4f} "
        f"max={logsumexp.max().item():.4f} mean={logsumexp.mean().item():.4f}"
    )


def check_blank_bias(
    modules: Tuple[torch.nn.Module, ...],
    base_batch: TensorDict,
    reference_lengths: torch.Tensor,
    target_query_budget: Optional[int],
    device: torch.device,
    bias: float,
) -> None:
    print(f"CHECK 2: injecting blank bias = {bias:+.2f}")
    modules_eval = tuple(module.eval() for module in modules)
    batch = _move_to_device(_clone_batch(base_batch), device)
    with torch.no_grad():
        outputs = _run_modules(modules_eval, batch, target_query_budget)
    logits = outputs.get("ctc_blank_logits")
    transition = outputs.get("ctc_blank_transition")
    lengths = outputs.get("decoder_lengths")
    if logits is None or transition is None or lengths is None:
        raise KeyError("Decoder outputs missing logits/transition/lengths for blank variant")

    ref_list = reference_lengths.detach().cpu().tolist()
    baseline_summary = _summarise_viterbi(
        outputs.get("viterbi_sequence", []),
        outputs.get("viterbi_collapsed_tokens", []),
        outputs.get("viterbi_rle_frames", []),
        ref_list,
    )
    _print_viterbi_summary("  Baseline Viterbi", baseline_summary)

    biased_logits = logits.clone()
    biased_logits[..., 0] += bias
    decoder: CTCCRFDecoder = modules_eval[-1]  # type: ignore[assignment]
    with torch.no_grad():
        (
            _seq,
            collapsed_tokens,
            _char_lengths,
            _frame_lengths,
            _texts,
            _rle_chars,
            rle_frames,
        ) = decoder._viterbi_blank(biased_logits, transition, lengths)
    biased_summary = _summarise_viterbi(_seq, collapsed_tokens, rle_frames, ref_list)
    _print_viterbi_summary("  After blank bias", biased_summary)


def overfit_single_segment(
    modules: Tuple[torch.nn.Module, ...],
    criterion: torch.nn.Module,
    base_batch: TensorDict,
    reference_lengths: torch.Tensor,
    target_query_budget: Optional[int],
    device: torch.device,
    steps: int,
    log_interval: int,
    lr: float,
) -> None:
    if steps <= 0:
        print("CHECK 3: skipped (overfit steps <= 0)")
        return

    print(f"CHECK 3: overfitting one segment for {steps} steps (lr={lr})")
    encoder, fusion, aggregator, decoder = modules
    encoder.train()
    fusion.train()
    aggregator.train()
    decoder.train()
    criterion.train()

    params: Iterable[torch.nn.Parameter] = (
        list(encoder.parameters())
        + list(fusion.parameters())
        + list(aggregator.parameters())
        + list(decoder.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    ref_list = reference_lengths.detach().cpu().tolist()

    for step in range(1, steps + 1):
        batch = _move_to_device(_clone_batch(base_batch), device)
        outputs = _run_modules(modules, batch, target_query_budget)
        loss = criterion(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == steps:
            with torch.no_grad():
                sequences = outputs.get("viterbi_sequence", [])
                collapsed = outputs.get("viterbi_collapsed_tokens", [])
                rle_frames = outputs.get("viterbi_rle_frames", [])
                summary = _summarise_viterbi(sequences, collapsed, rle_frames, ref_list)
                loss_value = float(loss.detach().cpu().item())
            print(f"  step={step:04d} loss={loss_value:.4f}")
            _print_viterbi_summary("    Viterbi", summary)

    for module in modules:
        module.eval()
    criterion.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blank CTC-CRF sanity checks")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "ctc_crf_blank_example.yaml"),
        help="Path to the experiment YAML configuration",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=0,
        help="Index of the segment inside the training split to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (e.g. 'cpu' or 'cuda:0')",
    )
    parser.add_argument(
        "--blank-bias",
        type=float,
        default=2.0,
        help="Constant bias added to the blank channel for CHECK 2",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=0,
        help="Number of optimisation steps for CHECK 3 (set 0 to skip)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Logging interval for CHECK 3",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate used during the single-segment overfit check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    base_batch, reference_lengths = _build_dataset(cfg, args.segment_index)
    modules = _instantiate_modules(cfg)
    modules = tuple(module.to(device=device) for module in modules)

    loss_cfg = cfg.losses.get("ctc_crf")
    if loss_cfg is None:
        raise ValueError("Loss configuration must contain a 'ctc_crf' entry")
    criterion = LOSS_REGISTRY.create(loss_cfg.name, **loss_cfg.params).to(device=device)

    target_query_budget = _infer_target_query_budget(base_batch)

    check_log_probabilities(modules, base_batch, target_query_budget, device)
    check_blank_bias(
        modules,
        base_batch,
        reference_lengths,
        target_query_budget,
        device,
        bias=args.blank_bias,
    )
    overfit_single_segment(
        modules,
        criterion,
        base_batch,
        reference_lengths,
        target_query_budget,
        device,
        steps=args.overfit_steps,
        log_interval=max(1, args.log_interval),
        lr=args.learning_rate,
    )


if __name__ == "__main__":
    main()