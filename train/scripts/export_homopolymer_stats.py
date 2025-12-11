"""Export central homopolymer statistics to a TSV file."""
from __future__ import annotations

import argparse
import csv
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import load_config
from ngpathfinder.losses.ctc_crf import _gather_reference
from ngpathfinder.modules.aggregator import AGGREGATOR_REGISTRY
from ngpathfinder.modules.decoder import DECODER_REGISTRY
from ngpathfinder.modules.decoder.rnnt import RNNTDecoder
from ngpathfinder.modules.encoder import ENCODER_REGISTRY
from ngpathfinder.modules.fusion import FUSION_REGISTRY
from ngpathfinder.utils.checkpoint import load_checkpoint
from ngpathfinder.utils.logging import configure_logging

from scripts.infer import (
    DecodeOptions,
    _build_torchaudio_decoder,
    _compact_padding_mask,
    _component_kwargs,
    _compute_alignment_stats,
    _compute_homopolymer_metrics,
    _ctc_greedy_decode,
    _ctc_prefix_beam_search,
    _decode_viterbi_sequence,
    _infer_target_query_budget,
    _load_module_states,
    _log_viterbi_stats,
    _make_dataloader,
    _move_to_device,
    _resolve_checkpoint,
    _resolve_dataset,
    _resolve_decode_options,
    _resolve_precision,
    _select_sequence_item,
    _tensor_to_base_string,
    _torchaudio_decode,
    _extract_valid_duration_logits,
    _extract_valid_ctc_logits,
    _rnnt_beam_decode_batch,
    _rnnt_greedy_decode_batch,
    _central_homopolymer_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and export central homopolymer statistics",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--ckpt",
        default="",
        help="Checkpoint path for model weights. Overrides config defaults when provided.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to run inference on (default: val)",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=("greedy", "beam", "torchaudio", "viterbi", "rnnt_greedy", "rnnt_beam"),
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
        "--output-name",
        default="central_homopolymer_stats.tsv",
        help="Name of the TSV file to create inside config.output_dir",
    )
    parser.add_argument(
        "--flank-bases",
        type=int,
        default=3,
        help="Number of context bases to retain on each side of the homopolymer run",
    )

    args = parser.parse_args()
    if args.flank_bases < 0:
        parser.error("--flank-bases must be non-negative")
    return args


def _homopolymer_bounds(mask: List[bool]) -> Optional[Tuple[int, int]]:
    try:
        start = mask.index(True)
    except ValueError:
        return None

    end = len(mask)
    while end > start and not mask[end - 1]:
        end -= 1
    return start, end


def _left_flank(sequence: str, start: int, flank_size: int) -> str:
    if flank_size <= 0:
        return ""

    flank = sequence[max(0, start - flank_size) : start]
    if len(flank) >= flank_size:
        return flank[-flank_size:]
    return ("N" * (flank_size - len(flank))) + flank


def _right_flank(sequence: str, end: int, flank_size: int) -> str:
    if flank_size <= 0:
        return ""

    flank = sequence[end : end + flank_size]
    if len(flank) >= flank_size:
        return flank[:flank_size]
    return flank + ("N" * (flank_size - len(flank)))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    configure_logging(cfg.output_dir)
    logger = logging.getLogger("export_homopolymer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision, use_autocast = _resolve_precision(cfg.trainer.precision, device)
    logger.info("Device: %s | Precision: %s", device, precision)

    flank_bases = int(args.flank_bases)
    flank_suffix = f"{flank_bases}mer" if flank_bases else "flank"
    left_flank_key = f"left_{flank_suffix}"
    right_flank_key = f"right_{flank_suffix}"
    logger.info(
        "Homopolymer flank context: %d base%s on each side",
        flank_bases,
        "" if flank_bases == 1 else "s",
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
    decoder = DECODER_REGISTRY.create(cfg.decoder.name, **_component_kwargs(cfg.decoder)).to(device)
    if decoder.__class__.__name__ == "CTCCRFDecoder" and not getattr(decoder, "return_viterbi", False):
        logger.info("Enabling Viterbi outputs for CTC-CRF decoder during inference")
        decoder.return_viterbi = True

    is_rnnt = isinstance(decoder, RNNTDecoder)

    if is_rnnt:
        if decode_options.rnnt_strategy == "beam":
            logger.info(
                "Decode strategy: rnnt_beam | beam_width=%d | max_symbols_per_step=%d",
                decode_options.beam_width,
                decode_options.rnnt_max_symbols_per_step,
            )
        else:
            logger.info(
                "Decode strategy: rnnt_greedy | max_symbols_per_step=%d",
                decode_options.rnnt_max_symbols_per_step,
            )

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

    results: List[Dict[str, Any]] = []
    skipped_segments = 0

    progress = tqdm(dataloader, desc=f"HP Export ({args.split})", unit="batch")
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

            decoded_sequences: Dict[int, str] = {}

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

            logits = outputs.get("ctc_logits")
            lengths = outputs.get("ctc_logit_lengths")
            gather_indices = outputs.get("ctc_logit_gather_indices")
            padding_mask = outputs.get("decoder_padding_mask")
            handled_batch = False

            if is_rnnt:
                handled_batch = True
                decode_fn = (
                    _rnnt_beam_decode_batch
                    if decode_options.rnnt_strategy == "beam"
                    else _rnnt_greedy_decode_batch
                )
                decoded_strings, frame_lengths, token_lengths = decode_fn(
                    decoder, outputs, decode_options, logger
                )
                for idx, seg_id in enumerate(segment_ids):
                    if idx >= len(decoded_strings):
                        break
                    decoded_sequences[idx] = decoded_strings[idx]

            if (
                not handled_batch
                and logits is not None
                and lengths is not None
                and decode_options.strategy != "viterbi"
            ):
                handled_batch = True
                valid_logits = _extract_valid_ctc_logits(
                    logits,
                    lengths,
                    gather_indices,
                    padding_mask,
                )
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
                    if valid <= 0:
                        decoded = ""
                    else:
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
                    decoded_sequences[idx] = decoded

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
                    decoded_sequences[idx] = decode_result.collapsed
                    _log_viterbi_stats(logger, seg_id, decode_result)

            if not handled_batch:
                logger.warning("Decoder outputs do not contain recognizable sequences; skipping batch")
                continue

            for idx, seg_id in enumerate(segment_ids):
                decoded = decoded_sequences.get(idx, "")
                if idx >= len(reference_targets_cpu):
                    logger.warning("No reference available for segment '%s'; skipping", seg_id)
                    skipped_segments += 1
                    continue

                target_len = (
                    int(reference_lengths_cpu[idx].item())
                    if idx < reference_lengths_cpu.numel()
                    else reference_targets_cpu[idx].numel()
                )
                if target_len <= 0:
                    logger.warning(
                        "Reference sequence for segment '%s' has non-positive length (%d); skipping",
                        seg_id,
                        target_len,
                    )
                    skipped_segments += 1
                    continue

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

                mask = _central_homopolymer_mask(reference_sequence)
                bounds = _homopolymer_bounds(mask)
                if bounds is None:
                    logger.warning(
                        "Segment '%s' does not contain a midpoint homopolymer run; skipping",
                        seg_id,
                    )
                    skipped_segments += 1
                    continue
                start, end = bounds
                actual_count = end - start
                if actual_count <= 0:
                    logger.warning(
                        "Homopolymer bounds invalid for segment '%s' (start=%d, end=%d); skipping",
                        seg_id,
                        start,
                        end,
                    )
                    skipped_segments += 1
                    continue

                alignment_stats = _compute_alignment_stats(reference_sequence, decoded)
                hp_metrics = _compute_homopolymer_metrics(
                    reference_sequence,
                    alignment_stats,
                    min_run=1,
                    mode="center",
                )
                if hp_metrics is None:
                    logger.warning(
                        "Unable to compute homopolymer metrics for segment '%s'; skipping",
                        seg_id,
                    )
                    skipped_segments += 1
                    continue

                predicted_count = int(hp_metrics.get("hyp_length", 0))
                left_flank = _left_flank(reference_sequence, start, flank_bases)
                right_flank = _right_flank(reference_sequence, end, flank_bases)

                results.append(
                    {
                        "segment": seg_id,
                        "predicted_count": predicted_count,
                        "reference_count": actual_count,
                        left_flank_key: left_flank,
                        right_flank_key: right_flank,
                    }
                )

    logger.info("Collected homopolymer statistics for %d segments", len(results))
    if skipped_segments:
        logger.info("Skipped %d segments without usable homopolymer statistics", skipped_segments)

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment",
                "predicted_count",
                "reference_count",
                left_flank_key,
                right_flank_key,
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(results)

    logger.info("Wrote homopolymer TSV to %s", output_path)


if __name__ == "__main__":
    main()