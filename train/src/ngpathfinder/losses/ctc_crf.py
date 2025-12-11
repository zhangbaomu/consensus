"""CTC-CRF negative log-likelihood implemented with k2."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from .base import LOSS_REGISTRY
from .ctc_common import _gather_reference, minimal_ctc_input_length
from ..utils.k2_ctc_crf import (
    DenseFsaInputs,
    build_blank_dense_inputs,
    build_blank_denominator_graph,
    build_ctc_blank_target_graph,
    build_dense_inputs,
    build_target_graph,
    build_transition_graph,
    create_fsa_vec,
    NUM_BASES,
    NUM_BLANK_CLASSES,
    NUM_STATES,
    STATE_EXTEND_OFFSET,
)

try:
    import k2
except ImportError as exc:  # pragma: no cover - guard for optional dependency
    raise ImportError(
        "The 'k2' package is required for the CTC-CRF loss. Install it via 'pip install k2'."
    ) from exc


class CTCCRFNegativeLogLikelihood(nn.Module):
    """Compute the CTC-CRF negative log likelihood using k2 dynamic programming."""

    def __init__(
        self,
        *,
        search_beam: float = 20.0,
        output_beam: float = 8.0,
        min_active_states: int = 1,
        max_active_states: int = 4096,
        use_double_scores: bool = False,
    ) -> None:
        super().__init__()
        self.search_beam = float(search_beam)
        self.output_beam = float(output_beam)
        self.min_active_states = int(min_active_states)
        self.max_active_states = int(max_active_states)
        self.use_double_scores = bool(use_double_scores)
        self._debug_calls = 0

    def _intersect_and_score(
        self,
        graphs: "k2.Fsa",
        dense: "k2.DenseFsaVec",
    ) -> Tensor:
        """Run k2 intersection with increasingly permissive beams."""

        search = max(self.search_beam, 1.0)
        output = max(self.output_beam, 1.0)
        max_states = max(self.max_active_states, self.min_active_states)

        for factor in (1.0, 2.0, 4.0):
            lattice = k2.intersect_dense_pruned(
                graphs,
                dense,
                search * factor,
                output * factor,
                self.min_active_states,
                int(max_states * factor),
            )
            scores = lattice.get_tot_scores(
                log_semiring=True,
                use_double_scores=self.use_double_scores,
            ).to(dtype=torch.float32)
            if torch.isfinite(scores).all():
                return scores

        lattice = k2.intersect_dense(graphs, dense, output_beam=output * 4.0)
        scores = lattice.get_tot_scores(
            log_semiring=True,
            use_double_scores=self.use_double_scores,
        ).to(dtype=torch.float32)
        return scores

    def _build_dense(self, emissions: Tensor, lengths: Tensor) -> DenseFsaInputs:
        return build_dense_inputs(emissions, lengths)

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        variant = outputs.get("ctc_variant", "ctc_crf_enter_extend")
        if variant == "ctc_crf_blank":
            return self._forward_blank(outputs, batch)
        if variant not in {"ctc_crf_enter_extend", "enter_extend", None}:
            raise ValueError(f"Unsupported CTC-CRF variant '{variant}'")
        return self._forward_enter_extend(outputs, batch)

    def _forward_enter_extend(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tensor:
        if "ctc_emissions" not in outputs:
            raise KeyError("Decoder outputs must contain 'ctc_emissions'")
        emissions = outputs["ctc_emissions"]
        if emissions.dim() != 3 or emissions.size(-1) != NUM_STATES:
            raise ValueError(
                "ctc_emissions must have shape (batch, time, 8) with enter/extend channels"
            )

        emissions_f32 = emissions.to(torch.float32)

        transition = outputs.get("ctc_transition")
        bos = outputs.get("ctc_bos")
        eos = outputs.get("ctc_eos")
        if transition is None or bos is None or eos is None:
            raise KeyError("Decoder outputs must include transition, bos and eos scores")

        transition = transition.to(device=emissions.device, dtype=torch.float32)
        bos = bos.to(device=emissions.device, dtype=torch.float32)
        eos = eos.to(device=emissions.device, dtype=torch.float32)

        raw_targets, _ = _gather_reference(batch)
        targets = [target.to(device=emissions.device, dtype=torch.long) for target in raw_targets]
        target_lengths = torch.tensor(
            [target.numel() for target in targets], device=emissions.device, dtype=torch.long
        )

        pad_mask = outputs.get("decoder_padding_mask")
        if pad_mask is None:
            pad_mask = batch.get("decoder_padding_mask")

        batch_size, time_steps, _ = emissions.shape
        if pad_mask is None:
            lengths = torch.full((batch_size,), time_steps, device=emissions.device, dtype=torch.long)
        else:
            if pad_mask.shape != emissions.shape[:2]:
                raise ValueError("decoder_padding_mask must have shape (batch, time) to match emissions")
            pad_mask_bool = pad_mask.to(torch.bool)
            lengths = (~pad_mask_bool).sum(dim=1).to(dtype=torch.long)
            if pad_mask_bool.any():
                emissions_f32 = emissions_f32.masked_fill(
                    pad_mask_bool.unsqueeze(-1), float("-inf")
                )
        dense_inputs = self._build_dense(emissions_f32, lengths)

        logz_den = emissions_f32.new_full((batch_size,), float("-inf"))
        logz_num = emissions_f32.new_full((batch_size,), float("-inf"))
        failure_messages = [""] * batch_size

        valid_time_mask = lengths > 0
        if valid_time_mask.any():
            valid_indices = valid_time_mask.nonzero(as_tuple=False).view(-1)
            dense_valid = dense_inputs.select(valid_indices)
            transition_graph = build_transition_graph(transition, bos, eos)
            denominator_graphs = [transition_graph.clone() for _ in range(valid_indices.numel())]
            denominator_graph = create_fsa_vec(denominator_graphs)
            subset_scores = self._intersect_and_score(
                denominator_graph, dense_valid.to_dense_fsa()
            )
            finite_mask = torch.isfinite(subset_scores)
            if finite_mask.any():
                valid_targets = valid_indices[finite_mask]
                logz_den[valid_targets.long()] = subset_scores[finite_mask]
            if (~finite_mask).any():
                bad_indices = valid_indices[(~finite_mask)].detach().cpu().tolist()
                for idx in bad_indices:
                    failure_messages[int(idx)] = "denominator_t0_all_paths_-inf"
        else:
            for idx in range(batch_size):
                failure_messages[idx] = "denominator_time_len<=0"

        zero_length_indices = (~valid_time_mask).nonzero(as_tuple=False).view(-1)
        for idx in zero_length_indices.detach().cpu().tolist():
            failure_messages[idx] = "time_len<=0"

        segment_ids = batch.get("segment_id")
        segment_list: List[str]
        if isinstance(segment_ids, list):
            segment_list = [str(seg) for seg in segment_ids]
        elif torch.is_tensor(segment_ids):
            segment_list = [
                str(item)
                for item in segment_ids.view(-1).detach().cpu().tolist()
            ]
        elif segment_ids is None:
            segment_list = []
        else:
            segment_list = [str(segment_ids)]

        numerator_graphs: List[k2.Fsa] = []
        numerator_indices: List[int] = []
        for idx, target in enumerate(targets):
            length = int(lengths[idx].item())
            target_len = int(target_lengths[idx].item())
            if length <= 0:
                failure_messages[idx] = "time_len<=0"
                continue
            if target_len == 0:
                failure_messages[idx] = "empty_target"
                continue
            if target_len > length:
                segment = segment_list[idx] if idx < len(segment_list) else str(idx)
                message = f"target_len={target_len} exceeds time_len={length}"
                logging.getLogger("ctc_crf").warning(
                    "ctc_crf numerator skipped segment %s: %s", segment, message
                )
                failure_messages[idx] = message
                continue
            numerator_graphs.append(build_target_graph(target, transition, bos, eos))
            numerator_indices.append(idx)

        if numerator_graphs:
            index_tensor = torch.tensor(numerator_indices, device=emissions.device, dtype=torch.long)
            dense_num = dense_inputs.select(index_tensor)
            numerator_graph = create_fsa_vec(numerator_graphs)
            subset_scores = self._intersect_and_score(
                numerator_graph, dense_num.to_dense_fsa()
            )
            logz_num[index_tensor] = subset_scores
        else:
            for idx, msg in enumerate(failure_messages):
                if not msg:
                    failure_messages[idx] = "numerator_graph_missing"

        valid = torch.isfinite(logz_num) & torch.isfinite(logz_den)
        reasons: List[str] = []
        for idx in range(batch_size):
            if valid[idx]:
                reasons.append("ok")
                continue
            reason = failure_messages[idx]
            if not reason:
                if not torch.isfinite(logz_den[idx]):
                    reason = "denominator_invalid"
                elif not torch.isfinite(logz_num[idx]):
                    reason = "numerator_invalid"
                else:
                    reason = "unknown"
            reasons.append(reason)

        valid_terms = logz_den[valid] - logz_num[valid]
        if valid_terms.numel() > 0:
            loss = valid_terms.mean()
        else:
            anchor = torch.nan_to_num(emissions_f32, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            loss = anchor + torch.tensor(1e6, device=emissions.device, dtype=torch.float32)

        invalid_ratio = (~valid).float().mean()
        if invalid_ratio > 0:
            loss = loss + invalid_ratio * 1e6

        logger = logging.getLogger("ctc_crf")
        self._debug_calls += 1
        valid_count = int(valid.sum().item())
        invalid_count = int((~valid).sum().item())
        should_report = (
            self._debug_calls <= 5
            or valid_terms.numel() == 0
            or invalid_ratio > 0
        )
        if should_report:
            logger.log(
                logging.WARNING if invalid_ratio > 0 else logging.INFO,
                "ctc_crf batch | segments=%s | valid=%d invalid=%d | loss=%.6f | invalid_ratio=%.4f | reasons=%s",
                segment_list if segment_list else batch.get("segment_id", []),
                valid_count,
                invalid_count,
                float(loss.detach().cpu().item()),
                float(invalid_ratio.detach().cpu().item()),
                reasons,
            )

        return loss.to(dtype=emissions.dtype)

    def _forward_blank(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tensor:
        logits = outputs.get("ctc_blank_logits")
        if logits is None:
            raise KeyError("Decoder outputs must contain 'ctc_blank_logits' for blank variant")
        if logits.dim() != 3 or logits.size(-1) != NUM_BLANK_CLASSES:
            raise ValueError(
                f"ctc_blank_logits must have shape (batch, time, {NUM_BLANK_CLASSES})"
            )

        transition = outputs.get("ctc_blank_transition")
        if transition is None:
            raise KeyError(
                "Decoder outputs must contain 'ctc_blank_transition' for blank variant"
            )

        logits_f32 = logits.to(torch.float32)
        transition_f32 = transition.to(device=logits.device, dtype=torch.float32)

        batch_size, time_steps, _ = logits.shape

        lengths = outputs.get("decoder_lengths")
        if lengths is None:
            lengths_tensor = torch.full(
                (batch_size,), time_steps, device=logits.device, dtype=torch.long
            )
        else:
            lengths_tensor = lengths.to(device=logits.device, dtype=torch.long)

        pad_mask = outputs.get("decoder_padding_mask")
        if pad_mask is None:
            pad_mask = batch.get("decoder_padding_mask")
        if pad_mask is not None:
            if pad_mask.shape != logits.shape[:2]:
                raise ValueError("decoder_padding_mask must match logits shape (batch, time)")
            pad_bool = pad_mask.to(torch.bool)
            logits_f32 = logits_f32.masked_fill(pad_bool.unsqueeze(-1), float("-inf"))
            lengths_tensor = (~pad_bool).sum(dim=1).to(dtype=torch.long, device=logits.device)

        dense_inputs = build_blank_dense_inputs(logits_f32, lengths_tensor)

        logz_den = logits_f32.new_full((batch_size,), float("-inf"))
        logz_num = logits_f32.new_full((batch_size,), float("-inf"))
        failure_messages = [""] * batch_size

        valid_time_mask = lengths_tensor > 0
        if valid_time_mask.any():
            valid_indices = valid_time_mask.nonzero(as_tuple=False).view(-1)
            dense_valid = dense_inputs.select(valid_indices)
            transition_graph = build_blank_denominator_graph(transition_f32)
            denominator_graphs = [transition_graph.clone() for _ in range(valid_indices.numel())]
            denominator_vec = create_fsa_vec(denominator_graphs)
            subset_scores = self._intersect_and_score(
                denominator_vec, dense_valid.to_dense_fsa()
            )
            finite_mask = torch.isfinite(subset_scores)
            if finite_mask.any():
                valid_targets = valid_indices[finite_mask]
                logz_den[valid_targets.long()] = subset_scores[finite_mask]
            if (~finite_mask).any():
                bad_indices = valid_indices[(~finite_mask)].detach().cpu().tolist()
                for idx in bad_indices:
                    failure_messages[int(idx)] = "denominator_t0_all_paths_-inf"
        else:
            for idx in range(batch_size):
                failure_messages[idx] = "denominator_time_len<=0"

        zero_length_indices = (~valid_time_mask).nonzero(as_tuple=False).view(-1)
        for idx in zero_length_indices.detach().cpu().tolist():
            failure_messages[idx] = "time_len<=0"

        raw_targets, _ = _gather_reference(batch)
        targets = [target.to(device=logits.device, dtype=torch.long) for target in raw_targets]
        target_lengths = torch.tensor(
            [target.numel() for target in targets], device=logits.device, dtype=torch.long
        )

        segment_ids = batch.get("segment_id")
        segment_list: List[str]
        if isinstance(segment_ids, list):
            segment_list = [str(seg) for seg in segment_ids]
        elif torch.is_tensor(segment_ids):
            segment_list = [
                str(item)
                for item in segment_ids.view(-1).detach().cpu().tolist()
            ]
        elif segment_ids is None:
            segment_list = []
        else:
            segment_list = [str(segment_ids)]

        minimal_lengths: List[int] = []
        insufficient: List[int] = []
        for idx, (target, tgt_len) in enumerate(zip(targets, target_lengths.tolist())):
            minimal = minimal_ctc_input_length(target, tgt_len)
            minimal_lengths.append(minimal)
            available = int(lengths_tensor[idx].item()) if idx < lengths_tensor.numel() else 0
            if minimal > available:
                insufficient.append(idx)

        if insufficient:
            details: List[str] = []
            for idx in insufficient[:8]:
                required = minimal_lengths[idx]
                available = int(lengths_tensor[idx].item())
                segment = "?"
                if idx < len(segment_list):
                    segment = str(segment_list[idx])
                else:
                    segment = str(idx)
                details.append(f"{segment}:required={required}>available={available}")

            raise RuntimeError(
                "CTC-CRF blank variant detected decoder budgets that are insufficient for the "
                "reference. Increase fusion.num_queries / dynamic_query_cap or relax blank "
                f"settings. offending_segments={details}"
            )

        numerator_graphs: List[k2.Fsa] = []
        numerator_indices: List[int] = []
        for idx, target in enumerate(targets):
            length = int(lengths_tensor[idx].item())
            target_len = int(target_lengths[idx].item())
            if length <= 0:
                failure_messages[idx] = "time_len<=0"
                continue
            if target_len == 0:
                failure_messages[idx] = "empty_target"
                continue
            if target_len > length:
                segment = segment_list[idx] if idx < len(segment_list) else str(idx)
                message = f"target_len={target_len} exceeds time_len={length}"
                logging.getLogger("ctc_crf").warning(
                    "ctc_crf numerator skipped segment %s: %s", segment, message
                )
                failure_messages[idx] = message
                continue
            numerator_graphs.append(
                build_ctc_blank_target_graph(target, transition_f32)
            )
            numerator_indices.append(idx)

        if numerator_graphs:
            index_tensor = torch.tensor(
                numerator_indices, device=logits.device, dtype=torch.long
            )
            dense_num = dense_inputs.select(index_tensor)
            numerator_graph = create_fsa_vec(numerator_graphs)
            subset_scores = self._intersect_and_score(
                numerator_graph, dense_num.to_dense_fsa()
            )
            logz_num[index_tensor] = subset_scores
        else:
            for idx, msg in enumerate(failure_messages):
                if not msg:
                    failure_messages[idx] = "numerator_graph_missing"

        valid = torch.isfinite(logz_num) & torch.isfinite(logz_den)
        reasons: List[str] = []
        for idx in range(batch_size):
            if valid[idx]:
                reasons.append("ok")
                continue
            reason = failure_messages[idx]
            if not reason:
                if not torch.isfinite(logz_den[idx]):
                    reason = "denominator_invalid"
                elif not torch.isfinite(logz_num[idx]):
                    reason = "numerator_invalid"
                else:
                    reason = "unknown"
            reasons.append(reason)

        valid_terms = logz_den[valid] - logz_num[valid]
        if valid_terms.numel() > 0:
            loss = valid_terms.mean()
        else:
            anchor = torch.nan_to_num(logits_f32, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            loss = anchor + torch.tensor(1e6, device=logits.device, dtype=torch.float32)

        invalid_ratio = (~valid).float().mean()
        if invalid_ratio > 0:
            loss = loss + invalid_ratio * 1e6

        logger = logging.getLogger("ctc_crf")
        self._debug_calls += 1
        valid_count = int(valid.sum().item())
        invalid_count = int((~valid).sum().item())
        should_report = (
            self._debug_calls <= 5
            or valid_terms.numel() == 0
            or invalid_ratio > 0
        )
        if should_report:
            logger.log(
                logging.WARNING if invalid_ratio > 0 else logging.INFO,
                "ctc_crf blank batch | segments=%s | valid=%d invalid=%d | loss=%.6f | invalid_ratio=%.4f | reasons=%s",
                segment_list if segment_list else batch.get("segment_id", []),
                valid_count,
                invalid_count,
                float(loss.detach().cpu().item()),
                float(invalid_ratio.detach().cpu().item()),
                reasons,
            )

        return loss.to(dtype=logits.dtype)


@LOSS_REGISTRY.register("ctc_crf")
def build_ctc_crf_loss(**kwargs: object) -> nn.Module:
    return CTCCRFNegativeLogLikelihood(**kwargs)


__all__ = [
    "CTCCRFNegativeLogLikelihood",
    "NUM_BASES",
    "NUM_STATES",
    "STATE_EXTEND_OFFSET",
    "minimal_ctc_input_length",
    "_gather_reference",
]