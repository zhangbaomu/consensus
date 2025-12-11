"""Infer a single segment and report detailed metrics."""
from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ngpathfinder.config import ComponentConfig, load_config
from ngpathfinder.data import ReadDataset, collate_read_batch
from ngpathfinder.losses.ctc_crf import _gather_reference
from ngpathfinder.losses.ctc_fast import minimal_ctc_input_length
from ngpathfinder.modules.aggregator import AGGREGATOR_REGISTRY
from ngpathfinder.modules.decoder import DECODER_REGISTRY
from ngpathfinder.modules.encoder import ENCODER_REGISTRY
from ngpathfinder.modules.fusion import FUSION_REGISTRY
from ngpathfinder.utils.checkpoint import load_checkpoint

# Reuse helper utilities from the multi-segment inference script.
from infer import (  # type: ignore[import]
    DecodeOptions,
    _compute_alignment_stats,
    _ctc_prefix_beam_search,
    _ctc_greedy_decode,
    _extract_valid_duration_logits,
    _extract_valid_ctc_logits,
    _compact_padding_mask,
    _dataset_kwargs,
    _infer_target_query_budget,
    _move_to_device,
    _resolve_precision,
    _resolve_decode_options,
    _tensor_to_base_string,
    _build_torchaudio_decoder,
    _torchaudio_decode,
    _decode_viterbi_sequence,
    _log_viterbi_stats,
    _select_sequence_item,
    _rnnt_beam_decode_batch,
    _rnnt_greedy_decode_batch,
    BASE_VOCAB,
)
from ngpathfinder.modules.decoder.rnnt import RNNTDecoder


def _expand_rle_frames_to_string(rle_frames: Any) -> Optional[str]:
    """Expand a frame-level RLE sequence (including blanks) into a string."""

    if not rle_frames:
        return None

    parts: List[str] = []
    for entry in rle_frames:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)):
            if len(entry) < 2:
                continue
            char, length = entry[0], entry[1]
        else:
            # Unexpected container; skip to stay robust during inference.
            continue

        try:
            count = int(length)
        except (TypeError, ValueError):
            continue

        if count <= 0:
            continue

        parts.append(str(char) * count)

    if not parts:
        return None

    return "".join(parts)


def _component_kwargs(component: ComponentConfig) -> Dict[str, Any]:
    params = dict(component.params)
    if component.variant:
        params.setdefault("variant", component.variant)
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single segment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration")
    parser.add_argument("--ckpt", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--segment", required=True, help="Path to the target segment directory")
    parser.add_argument(
        "--precision",
        default=None,
        help="Optional precision override (float32, float16, bfloat16). Defaults to config value.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional log file path. If omitted, results are printed to the terminal only.",
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
        help="Override number of torchaudio hypotheses to retain",
    )
    parser.add_argument(
        "--torchaudio-beam-threshold",
        type=float,
        default=None,
        help="Override torchaudio beam threshold (score difference pruning)",
    )
    return parser.parse_args()


def _setup_logging(output_path: Optional[str]) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )

    return logging.getLogger("infer_one")


def _load_modules(cfg: Any, device: torch.device) -> Dict[str, torch.nn.Module]:
    encoder = ENCODER_REGISTRY.create(cfg.encoder.name, **_component_kwargs(cfg.encoder)).to(device)
    fusion = FUSION_REGISTRY.create(cfg.fusion.name, **_component_kwargs(cfg.fusion)).to(device)
    aggregator = AGGREGATOR_REGISTRY.create(
        cfg.aggregator.name, **_component_kwargs(cfg.aggregator)
    ).to(device)
    decoder = DECODER_REGISTRY.create(
        cfg.decoder.name, **_component_kwargs(cfg.decoder)
    ).to(device)

    if decoder.__class__.__name__ == "CTCCRFDecoder" and not getattr(decoder, "return_viterbi", False):
        decoder.return_viterbi = True

    for module in (encoder, fusion, aggregator, decoder):
        module.eval()

    return {
        "encoder": encoder,
        "fusion": fusion,
        "aggregator": aggregator,
        "decoder": decoder,
    }


def _load_checkpoint_state(path: Path, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist")
    logger.info("Loading checkpoint from %s", path)
    state = load_checkpoint(str(path), map_location=device)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint file does not contain a state dictionary")
    return state


def _apply_checkpoint(
    modules: Dict[str, torch.nn.Module], state: Dict[str, Any], logger: logging.Logger
) -> None:
    for name, module in modules.items():
        module_state = state.get(name)
        if module_state is None:
            logger.warning("Checkpoint is missing '%s' parameters; leaving module unchanged", name)
            continue
        module.load_state_dict(module_state, strict=False)
        logger.info("Loaded parameters for %s", name)


def _run_model(
    batch: Dict[str, Any],
    modules: Dict[str, torch.nn.Module],
    device: torch.device,
    precision: torch.dtype,
    use_autocast: bool,
    target_query_budget: Optional[int],
) -> Dict[str, torch.Tensor]:
    inferred_budget = target_query_budget
    if inferred_budget is None:
        inferred_budget = _infer_target_query_budget(batch)
    encoder = modules["encoder"]
    fusion = modules["fusion"]
    aggregator = modules["aggregator"]
    decoder = modules["decoder"]

    if use_autocast:
        autocast_cm = torch.autocast(device_type=device.type, dtype=precision)
    else:
        autocast_cm = nullcontext()

    with autocast_cm:
        encoded = encoder(batch)
        if inferred_budget is not None:
            base_queries = int(getattr(fusion, "num_queries", inferred_budget))
            encoded["target_query_count"] = max(base_queries, inferred_budget)
        fused = fusion(encoded)
        aggregated = aggregator(fused)
        if "decoder_padding_mask" not in aggregated:
            pad_mask = batch.get("decoder_padding_mask")
            if pad_mask is not None:
                aggregated["decoder_padding_mask"] = pad_mask
        outputs = decoder(aggregated)

    return outputs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger = _setup_logging(args.output)

    if os.getenv("PF2_ALLOW_CTC_OVERFLOW") == "1":
        logger.warning(
            "Environment variable PF2_ALLOW_CTC_OVERFLOW=1 detected. Training with this flag "
            "suppresses length overflow checks and can lead to degenerate all-blank predictions."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_setting = args.precision or cfg.trainer.precision
    precision, use_autocast = _resolve_precision(precision_setting, device)
    logger.info("Device: %s | Precision: %s", device, precision)

    torchaudio_overrides: Dict[str, Any] = {}
    if args.torchaudio_nbest is not None:
        torchaudio_overrides["nbest"] = args.torchaudio_nbest
    if args.torchaudio_beam_threshold is not None:
        torchaudio_overrides["beam_threshold"] = args.torchaudio_beam_threshold

    decode_options: DecodeOptions = _resolve_decode_options(
        cfg,
        logger,
        strategy_override=args.decode_strategy,
        beam_width_override=args.beam_width,
        prune_override=args.beam_prune_threshold,
        torchaudio_overrides=torchaudio_overrides or None,
    )
    torchaudio_decoder = None

    modules = _load_modules(cfg, device)
    fusion_module = modules["fusion"]
    logger.info(
        "Fusion query settings | num_queries=%s | max_learned=%s | dynamic_cap=%s",
        getattr(fusion_module, "num_queries", "n/a"),
        getattr(fusion_module, "max_learned_queries", "n/a"),
        getattr(fusion_module, "dynamic_query_cap", "n/a"),
    )

    decoder_module = modules["decoder"]
    decoder_class = decoder_module.__class__.__name__
    is_rnnt = isinstance(decoder_module, RNNTDecoder)

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
    elif decode_options.strategy == "beam":
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
                rnnt_strategy=decode_options.rnnt_strategy,
                rnnt_max_symbols_per_step=decode_options.rnnt_max_symbols_per_step,
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

    checkpoint_path = Path(args.ckpt)
    checkpoint_state = _load_checkpoint_state(checkpoint_path, device, logger)
    _apply_checkpoint(modules, checkpoint_state, logger)

    data_params = cfg.data.params if hasattr(cfg.data, "params") else {}
    if not isinstance(data_params, dict):
        data_params = {}
    dataset_kwargs = _dataset_kwargs(data_params)
    segment_path = Path(args.segment)
    dataset = ReadDataset(segment_path, **dataset_kwargs)

    if len(dataset) != 1:
        logger.warning(
            "Segment path '%s' resolved to %d segments; proceeding with the first entry",
            segment_path,
            len(dataset),
        )

    sample = dataset[0]
    batch = collate_read_batch([sample])
    batch = _move_to_device(batch, device)

    target_query_budget = _infer_target_query_budget(batch)
    if target_query_budget is not None:
        logger.info("Inferred target query budget from reference: %d", target_query_budget)
    else:
        logger.warning("Failed to infer target query budget; fallback to fusion defaults")

    with torch.no_grad():
        outputs = _run_model(
            batch,
            modules,
            device,
            precision,
            use_autocast,
            target_query_budget,
        )

    is_ctc_crf = decoder_class == "CTCCRFDecoder"

    logits = outputs.get("ctc_logits")
    lengths = outputs.get("ctc_logit_lengths")
    gather_indices = outputs.get("ctc_logit_gather_indices")
    padding_mask = outputs.get("decoder_padding_mask")
    sequences = outputs.get("viterbi_sequence") if is_ctc_crf else None
    embedding = outputs.get("embedding")
    char_run_lengths_out = outputs.get("viterbi_char_run_length") if is_ctc_crf else None
    frame_run_lengths_out = outputs.get("viterbi_frame_run_length") if is_ctc_crf else None
    collapsed_tokens_out = outputs.get("viterbi_collapsed_tokens") if is_ctc_crf else None
    emissions = outputs.get("ctc_emissions") if is_ctc_crf else None

    decoded_sequence = ""
    valid_length = 0
    non_blank = 0
    collapsed_sequence = ""
    collapsed_len = 0
    expanded_sequence: Optional[str] = None
    expanded_len: Optional[int] = None
    frame_total_val: Optional[int] = None
    same_base_reenter = 0
    run_total = 0
    raw_argmax_sequence = ""
    blank_viterbi_sequence: Optional[str] = None

    best_indices = None
    length_tensor = None

    duration_priors_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if is_rnnt:
        decoded_strings: List[str]
        frame_lengths: List[int]
        token_lengths: List[int]
        decode_fn = (
            _rnnt_beam_decode_batch
            if decode_options.rnnt_strategy == "beam"
            else _rnnt_greedy_decode_batch
        )
        decoded_strings, frame_lengths, token_lengths = decode_fn(
            decoder_module, outputs, decode_options, logger
        )
        if decoded_strings:
            decoded_sequence = decoded_strings[0]
            collapsed_sequence = decoded_sequence
            expanded_sequence = decoded_sequence
            collapsed_len = len(decoded_sequence)
            expanded_len = collapsed_len
            run_total = token_lengths[0] if token_lengths else collapsed_len
            non_blank = run_total
            valid_length = frame_lengths[0] if frame_lengths else collapsed_len
            frame_total_val = valid_length
        else:
            logger.warning("RNNT decoding produced no hypotheses; treating output as empty")
    elif logits is not None and lengths is not None:
        valid_logits = _extract_valid_ctc_logits(logits, lengths, gather_indices, padding_mask)

        if decode_options.strategy == "beam" and decode_options.use_duration_prior:
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
    elif sequences is None:
        raise RuntimeError(
            "Decoder outputs missing both CTC logits and Viterbi sequence; cannot decode"
        )

    reference_targets, reference_lengths = _gather_reference(batch)

    reference_strings = batch.get("reference_string", [])
    reference_paths = batch.get("reference_path", [])
    reference_sequence = ""
    reference_path = ""
    if isinstance(reference_strings, list) and reference_strings:
        reference_sequence = str(reference_strings[0])
    if isinstance(reference_paths, list) and reference_paths:
        reference_path = str(reference_paths[0])

    if not reference_targets:
        raise ValueError(
            f"No reference targets were found for segment '{sample.get('segment_id', '')}'"
        )

    if not isinstance(reference_lengths, torch.Tensor) or reference_lengths.numel() == 0:
        raise ValueError(
            f"Reference length tensor is missing for segment '{sample.get('segment_id', '')}'"
        )

    reference_tensor = reference_targets[0].detach().to("cpu")
    target_len = int(reference_lengths[0].item())
    if target_len <= 0:
        raise ValueError(
            f"Reference sequence for segment '{sample.get('segment_id', '')}' has non-positive length ({target_len})"
        )

    tensor_sequence = _tensor_to_base_string(reference_tensor[:target_len])
    if reference_sequence and reference_sequence != tensor_sequence:
        logger.warning(
            "Reference string mismatch between FASTA text and tensor representation; using FASTA value"
        )
    if not reference_sequence:
        reference_sequence = tensor_sequence

    if is_rnnt:
        # RNNT decoding already produced token/frame lengths; fall back to resolver if missing
        if valid_length <= 0:
            if frame_lengths and frame_lengths[0] > 0:
                valid_length = frame_lengths[0]
            else:
                length_source = outputs.get("rnnt_logit_lengths")
                if isinstance(length_source, torch.Tensor) and length_source.numel() > 0:
                    valid_length = int(length_source[0].item())
                elif isinstance(padding_mask, torch.Tensor) and padding_mask.numel() > 0:
                    target_steps = (
                        embedding.size(1)
                        if embedding is not None and embedding.dim() >= 2
                        else padding_mask.size(-1)
                    )
                    compact_mask = _compact_padding_mask(padding_mask, target_steps)
                    valid_mask = torch.logical_not(compact_mask)
                    valid_length = int(valid_mask.sum(dim=1).to("cpu").item())
        frame_total_val = valid_length if frame_total_val is None else frame_total_val
        non_blank = run_total
    elif (
        best_indices is not None
        and length_tensor is not None
        and decode_options.strategy != "viterbi"
    ):
        valid_length = (
            int(length_tensor[0].item()) if length_tensor.numel() else best_indices.size(1)
        )
        if decode_options.strategy == "beam" and valid_length > 0:
            log_prob_slice = torch.log_softmax(
                valid_logits[0, :valid_length].to(torch.float32), dim=-1
            )
            sample_priors = None
            if duration_priors_batch is not None:
                stay_batch, advance_batch = duration_priors_batch
                sample_priors = (stay_batch[0], advance_batch[0])
            decoded_sequence = _ctc_prefix_beam_search(
                log_prob_slice,
                valid_length=valid_length,
                beam_width=decode_options.beam_width,
                prune_threshold=decode_options.beam_prune_threshold,
                duration_priors=sample_priors,
            )
        elif decode_options.strategy == "torchaudio" and torchaudio_decoder is not None:
            log_prob_slice = torch.log_softmax(
                valid_logits[0, :valid_length].to(torch.float32), dim=-1
            )
            decoded_sequence = _torchaudio_decode(
                torchaudio_decoder, log_prob_slice, valid_length
            )
        elif decode_options.strategy == "beam":
            decoded_sequence = ""
        else:
            decoded_sequence = _ctc_greedy_decode(best_indices[0], valid_length)
        raw_tokens = best_indices[0][:valid_length].tolist()
        raw_argmax_sequence = "".join(
            "-" if token == 0 else BASE_VOCAB.get(token, "N") for token in raw_tokens
        )
        non_blank = sum(1 for token in raw_tokens if token != 0)
        collapsed_sequence = decoded_sequence
        collapsed_len = len(decoded_sequence)
        expanded_sequence = decoded_sequence
        expanded_len = len(decoded_sequence)
        run_total = expanded_len
    else:
        if decode_options.strategy not in {"greedy", "viterbi"}:
            logger.warning(
                "Decode strategy '%s' is not supported for CTC-CRF outputs; using Viterbi results",
                decode_options.strategy,
            )
        if sequences is None or not sequences:
            decoded_sequence = ""
            valid_length = 0
            non_blank = 0
            collapsed_sequence = ""
            collapsed_len = 0
        else:
            decode_result = _decode_viterbi_sequence(
                sequences[0],
                char_run_lengths=_select_sequence_item(char_run_lengths_out, 0),
                frame_run_lengths=_select_sequence_item(frame_run_lengths_out, 0),
                collapsed_tokens=_select_sequence_item(collapsed_tokens_out, 0),
            )
            collapsed_sequence = decode_result.collapsed
            collapsed_len = decode_result.collapsed_length
            expanded_sequence = decode_result.expanded
            expanded_len = decode_result.expanded_length
            decoded_sequence = collapsed_sequence
            rle_frames = None
            frame_rle_source = outputs.get("viterbi_rle_frames")
            if isinstance(frame_rle_source, list) and frame_rle_source:
                rle_frames = frame_rle_source[0]
            elif torch.is_tensor(frame_rle_source) and frame_rle_source.dim() > 0:
                rle_frames = frame_rle_source[0].tolist()

            if rle_frames is not None:
                blank_viterbi_sequence = _expand_rle_frames_to_string(rle_frames)

            stats = _log_viterbi_stats(logger, sample.get("segment_id", ""), decode_result)
            char_total = int(stats["char_total"])
            frame_total = int(stats["frame_total"])
            same_base_reenter = int(stats["same_base_reenter"])
            run_total = char_total
            frame_total_val = frame_total

            if frame_total > 0:
                valid_length = frame_total
            else:
                mask_length = None
                if isinstance(padding_mask, torch.Tensor) and padding_mask.numel() > 0:
                    target_steps = (
                        emissions.size(1)
                        if emissions is not None and emissions.dim() >= 2
                        else padding_mask.size(-1)
                    )
                    compact_mask = _compact_padding_mask(padding_mask, target_steps)
                    valid_mask = torch.logical_not(compact_mask)
                    mask_vals = valid_mask.sum(dim=1).to("cpu").tolist()
                    if mask_vals:
                        mask_length = int(mask_vals[0])
                if mask_length is None and emissions is not None and emissions.dim() >= 2:
                    mask_length = int(emissions.size(1))
                if mask_length is None or mask_length < collapsed_len:
                    mask_length = max(collapsed_len, run_total)
                valid_length = mask_length
        non_blank = run_total

    logger.info("Segment: %s", sample.get("segment_id", ""))
    logger.info(
        "Decoded summary | segments=1 | decoded_len=%d | char_total=%d | frame_total=%s | "
        "logit_len=%d | non_blank=%d | same_base_reenter=%d",
        len(decoded_sequence),
        int(run_total) if isinstance(run_total, int) else len(decoded_sequence),
        str(frame_total_val) if frame_total_val is not None else "n/a",
        valid_length,
        non_blank,
        same_base_reenter,
    )

    minimal_required = 0
    if reference_targets and target_len > 0:
        if is_rnnt:
            minimal_required = target_len
        else:
            minimal_required = minimal_ctc_input_length(reference_targets[0], target_len)
    logger.info(
        "Decoder budget vs reference | active_queries=%d | target_tokens=%d | minimal_required=%d",
        valid_length,
        target_len,
        minimal_required,
    )

    decoder_mask = outputs.get("decoder_padding_mask")
    if isinstance(decoder_mask, torch.Tensor) and decoder_mask.numel() > 0:
        active = (~decoder_mask[0].to(torch.bool)).sum().item()
        logger.info("Decoder padding mask active positions (sample 0): %d", int(active))

    if raw_argmax_sequence and valid_length > 0:
        logger.info(
            "Raw argmax sequence with blanks (len=%d): %s",
            valid_length,
            raw_argmax_sequence,
        )

    if blank_viterbi_sequence:
        logger.info(
            "Viterbi sequence with blanks (len=%d): %s",
            len(blank_viterbi_sequence),
            blank_viterbi_sequence,
        )

    if best_indices is not None and logits is not None and valid_length > 0:
        blank_ratio = 1.0 - (non_blank / float(valid_length))
        logger.info("Argmax blank ratio: %.2f%%", blank_ratio * 100.0)

        step_probs = torch.softmax(logits[0, :valid_length].to(torch.float32), dim=-1)
        blank_probs = step_probs[:, 0]
        high_blank = (blank_probs > 0.95).float().mean().item() * 100.0
        logger.info(
            "Blank probability stats | mean=%.3f | min=%.3f | max=%.3f | >0.95=%.2f%%",
            float(blank_probs.mean().item()),
            float(blank_probs.min().item()),
            float(blank_probs.max().item()),
            high_blank,
        )

        if step_probs.size(-1) > 1:
            top_non_blank = step_probs[:, 1:].max(dim=-1).values
            logger.info(
                "Top non-blank probability stats | mean=%.3f | max=%.3f",
                float(top_non_blank.mean().item()),
                float(top_non_blank.max().item()),
            )

    if reference_path:
        logger.info("Reference FASTA path: %s", reference_path)

    if reference_sequence and target_len > 0:
        alignment = _compute_alignment_stats(reference_sequence, decoded_sequence)
        acc_aligned = (
            alignment.matches / float(alignment.alignment_length)
            if alignment.alignment_length > 0
            else 1.0
        )
        acc_ref = (
            alignment.matches / float(alignment.reference_length)
            if alignment.reference_length > 0
            else 1.0
        )
        norm_denom = max(alignment.reference_length, len(decoded_sequence))
        normalized_distance = (
            alignment.distance / float(norm_denom) if norm_denom > 0 else 0.0
        )

        logger.info("Evaluation metrics for the segment:")
        logger.info("  Acc@aligned: %.4f", acc_aligned)
        logger.info("  Base-level accuracy (alignment-aware): %.4f", acc_aligned)
        logger.info("  Acc@pos (reference): %.4f", acc_ref)
        logger.info("  Levenshtein distance: %.4f", alignment.distance)
        logger.info("  Normalized Levenshtein distance: %.4f", normalized_distance)
    else:
        logger.info("Reference sequence unavailable; skipping accuracy metrics")

    logger.info("Reference sequence (len=%d): %s", target_len, reference_sequence or "<empty>")
    logger.info(
        "Collapsed decoded sequence (len=%d): %s",
        collapsed_len,
        collapsed_sequence or "<empty>",
    )
    if expanded_sequence is not None and expanded_sequence != collapsed_sequence:
        logger.info(
            "Expanded decoded sequence  (len=%d): %s",
            expanded_len if expanded_len is not None else len(expanded_sequence),
            expanded_sequence or "<empty>",
        )


if __name__ == "__main__":
    main()