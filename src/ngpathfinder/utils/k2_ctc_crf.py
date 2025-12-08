"""Helper utilities for K2-based CTC-CRF graphs."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import List, Optional, Sequence

import torch

from ..losses.ctc_constants import NUM_BASES, NUM_BLANK_CLASSES, NUM_STATES, STATE_EXTEND_OFFSET


def _require_k2() -> ModuleType:
    try:
        import k2  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard for optional dependency
        raise ImportError(
            "The 'k2' package is required for K2-based CTC-CRF components. "
            "Install it via 'pip install k2'."
        ) from exc
    return k2  # type: ignore


def _require_k2_utils() -> ModuleType:
    module = _require_k2()
    utils = getattr(module, "utils", None)
    if utils is None:  # pragma: no cover - defensive guard
        raise ImportError("k2.utils is unavailable; install a compatible k2 build")
    return utils

_LABEL_OFFSET = 1  # reserve 0 for epsilon
BLANK_LABEL = 1


@dataclass(frozen=True)
class DenseFsaInputs:
    """Container bundling the dense representation passed to k2."""

    scores: torch.Tensor
    supervision_segments: torch.Tensor

    def to_dense_fsa(self) -> "k2.DenseFsaVec":
        k2 = _require_k2()
        scores = self.scores
        if not scores.is_contiguous():
            scores = scores.contiguous()
        supervision = self.supervision_segments
        if supervision.device.type != "cpu":
            supervision = supervision.to(device=torch.device("cpu"))
        if not supervision.is_contiguous():
            supervision = supervision.contiguous()
        return k2.DenseFsaVec(scores, supervision)

    def select(self, indices: torch.Tensor) -> "DenseFsaInputs":
        if indices.numel() == 0:
            raise ValueError("indices must contain at least one element")
        indices = indices.to(dtype=torch.long, device=self.scores.device)
        scores = torch.index_select(self.scores, 0, indices)
        supervision_indices = indices.to(device=self.supervision_segments.device, dtype=torch.long)
        supervision = torch.index_select(
            self.supervision_segments, 0, supervision_indices
        ).clone()
        supervision[:, 0] = torch.arange(
            indices.numel(), device=supervision.device, dtype=torch.int32
        )
        return DenseFsaInputs(scores=scores, supervision_segments=supervision)


def build_dense_inputs(emissions: torch.Tensor, lengths: torch.Tensor) -> DenseFsaInputs:
    """Create a :class:`DenseFsaInputs` object from emission logits.

    The returned `scores` tensor prepends an epsilon channel so that state
    indices can be used directly as k2 labels via ``label = state + 1``.
    """

    if emissions.dim() != 3:
        raise ValueError("emissions must have shape (batch, time, state_dim)")

    batch, time_steps, state_dim = emissions.shape
    if state_dim != NUM_STATES:
        raise ValueError(
            f"emissions expect {NUM_STATES} channels with enter/extend ordering"
        )

    if lengths.dim() != 1 or lengths.numel() != batch:
        raise ValueError("lengths must be a 1D tensor aligned with the batch")

    device = emissions.device
    dtype = emissions.dtype

    if dtype not in (torch.float32, torch.float64):
        emissions = emissions.to(device=device, dtype=torch.float32)
        dtype = emissions.dtype

    blank_channel = emissions.new_full((batch, time_steps, 1), float("-inf"))
    scores = torch.cat([blank_channel, emissions], dim=-1)

    supervision_segments = torch.stack(
        [
            torch.arange(batch, device=torch.device("cpu"), dtype=torch.int32),
            torch.zeros(batch, device=torch.device("cpu"), dtype=torch.int32),
            lengths.to(device=torch.device("cpu"), dtype=torch.int32),
        ],
        dim=1,
    )
    return DenseFsaInputs(scores=scores, supervision_segments=supervision_segments)


def _add_arc(
    sources: List[int],
    dests: List[int],
    labels: List[int],
    weights: List[torch.Tensor],
    *,
    src: int,
    dest: int,
    label: int,
    score: torch.Tensor,
) -> None:
    if not torch.isfinite(score):
        return
    sources.append(src)
    dests.append(dest)
    labels.append(label)
    weights.append(score)


def build_transition_graph(
    transition: torch.Tensor,
    bos: torch.Tensor,
    eos: torch.Tensor,
) -> "k2.Fsa":
    """Construct the denominator transition graph shared across the batch."""

    k2 = _require_k2()
    if transition.shape != (NUM_STATES, NUM_STATES):
        raise ValueError("transition must have shape (8, 8) for enter/extend states")

    sources: List[int] = []
    dests: List[int] = []
    labels: List[int] = []
    scores: List[torch.Tensor] = []

    start_state = 0
    final_state = NUM_STATES + 1

    for state in range(NUM_STATES):
        label = state + _LABEL_OFFSET
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=start_state,
            dest=state + 1,
            label=label,
            score=bos[state],
        )

    for src_state in range(NUM_STATES):
        for dst_state in range(NUM_STATES):
            label = dst_state + _LABEL_OFFSET
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=src_state + 1,
                dest=dst_state + 1,
                label=label,
                score=transition[src_state, dst_state],
            )

    for state in range(NUM_STATES):
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=state + 1,
            dest=final_state,
            label=-1,
            score=eos[state],
        )

    if not sources:
        raise RuntimeError("transition graph contains no finite arcs")

    device = transition.device
    arcs_tensor = torch.stack(
        [
            torch.tensor(sources, dtype=torch.int32, device=device),
            torch.tensor(dests, dtype=torch.int32, device=device),
            torch.tensor(labels, dtype=torch.int32, device=device),
            torch.zeros(len(sources), dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    sort_key = arcs_tensor[:, 0] * (final_state + 1) + arcs_tensor[:, 1]
    order = torch.argsort(sort_key)
    arcs_sorted = arcs_tensor.index_select(0, order)
    score_tensor = torch.stack(scores, dim=0).to(dtype=torch.float32, device=device)
    score_tensor = score_tensor.index_select(0, order)

    fsa = k2.Fsa(arcs_sorted.cpu())
    fsa.scores = score_tensor.to(device=torch.device("cpu"))

    if device.type != "cpu":
        fsa = fsa.to(device)
        fsa.scores = score_tensor.to(device)
    return k2.arc_sort(fsa)


def build_blank_dense_inputs(logits: torch.Tensor, lengths: torch.Tensor) -> DenseFsaInputs:
    """Prepare dense inputs for the blank-variant CTC-CRF graphs."""

    if logits.dim() != 3:
        raise ValueError("logits must have shape (batch, time, classes)")

    batch, time_steps, num_classes = logits.shape
    if num_classes != NUM_BLANK_CLASSES:
        raise ValueError(
            f"blank-variant expects {NUM_BLANK_CLASSES} emission classes including blank"
        )

    if lengths.dim() != 1 or lengths.numel() != batch:
        raise ValueError("lengths must be a 1D tensor aligned with the batch")

    scores = logits.to(dtype=torch.float32)
    epsilon = torch.zeros(batch, time_steps, 1, device=scores.device, dtype=torch.float32)
    scores = torch.cat([epsilon, scores], dim=-1)

    supervision_segments = torch.stack(
        [
            torch.arange(batch, device=torch.device("cpu"), dtype=torch.int32),
            torch.zeros(batch, device=torch.device("cpu"), dtype=torch.int32),
            lengths.to(device=torch.device("cpu"), dtype=torch.int32),
        ],
        dim=1,
    )
    return DenseFsaInputs(scores=scores, supervision_segments=supervision_segments)


def build_blank_denominator_graph(transition: torch.Tensor) -> "k2.Fsa":
    """Construct a denominator graph for blank-based CTC-CRF variants."""

    k2 = _require_k2()
    if transition.dim() != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition matrix must be square")
    if transition.shape[0] != NUM_BLANK_CLASSES:
        raise ValueError(
            f"transition matrix must have shape ({NUM_BLANK_CLASSES}, {NUM_BLANK_CLASSES})"
        )

    device = transition.device
    transition = transition.to(dtype=torch.float32)

    sources: List[int] = []
    dests: List[int] = []
    labels: List[int] = []
    scores: List[torch.Tensor] = []

    start_state = 0
    final_state = NUM_BLANK_CLASSES + 1

    zero = transition.new_zeros(())

    for label_idx in range(NUM_BLANK_CLASSES):
        label = label_idx + _LABEL_OFFSET
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=start_state,
            dest=label_idx + 1,
            label=label,
            score=zero,
        )

    for src_idx in range(NUM_BLANK_CLASSES):
        for dst_idx in range(NUM_BLANK_CLASSES):
            score = transition[src_idx, dst_idx]
            label = dst_idx + _LABEL_OFFSET
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=src_idx + 1,
                dest=dst_idx + 1,
                label=label,
                score=score,
            )

    for state_idx in range(NUM_BLANK_CLASSES):
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=state_idx + 1,
            dest=final_state,
            label=-1,
            score=zero,
        )

    if not sources:
        raise RuntimeError("blank denominator graph contains no finite arcs")

    arcs_tensor = torch.stack(
        [
            torch.tensor(sources, dtype=torch.int32, device=device),
            torch.tensor(dests, dtype=torch.int32, device=device),
            torch.tensor(labels, dtype=torch.int32, device=device),
            torch.zeros(len(sources), dtype=torch.int32, device=device),
        ],
        dim=1,
    )

    sort_key = arcs_tensor[:, 0] * (final_state + 1) + arcs_tensor[:, 1]
    order = torch.argsort(sort_key)
    arcs_sorted = arcs_tensor.index_select(0, order)

    score_tensor = torch.stack(scores, dim=0).to(dtype=torch.float32, device=device)
    score_tensor = score_tensor.index_select(0, order.to(device=score_tensor.device))

    fsa = k2.Fsa(arcs_sorted.cpu())
    fsa.scores = score_tensor.to(device=torch.device("cpu"))

    if device.type != "cpu":
        fsa = fsa.to(device)
        fsa.scores = score_tensor.to(device)
    return k2.arc_sort(fsa)


def build_ctc_blank_target_graph(
    target: torch.Tensor, transition: Optional[torch.Tensor] = None
) -> "k2.Fsa":
    """Create numerator graph adhering to CTC collapse rules with blanks.

    When ``transition`` is provided, its per-class scores are added to the arcs so
    that numerator and denominator terms share the same transition biases. The
    matrix must follow the blank-first ordering used by the decoder
    (``{blank, A, C, G, T}``).
    """

    k2 = _require_k2()
    if target.dim() != 1:
        raise ValueError("target must be a 1D tensor of base indices")
    if target.numel() == 0:
        raise ValueError("target sequence must be non-empty")
    if torch.any(target <= 0):
        raise ValueError("target indices must be strictly positive (1-based bases)")

    transition_scores: Optional[torch.Tensor]
    if transition is not None:
        if transition.dim() != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("transition matrix must be square")
        if transition.shape[0] != NUM_BLANK_CLASSES:
            raise ValueError(
                f"transition matrix must have shape ({NUM_BLANK_CLASSES}, {NUM_BLANK_CLASSES})"
            )
        transition_scores = transition.to(dtype=torch.float32)
        score_device = transition_scores.device
    else:
        transition_scores = None
        score_device = target.device

    base_indices = target.to(dtype=torch.long)
    extended: List[int] = [BLANK_LABEL]
    for token in base_indices.tolist():
        extended.append(int(token) + 1)
        extended.append(BLANK_LABEL)

    num_symbols = len(extended)
    start_state = 0
    final_state = num_symbols + 1

    sources: List[int] = []
    dests: List[int] = []
    labels: List[int] = []
    scores: List[torch.Tensor] = []

    zero = torch.zeros((), device=score_device, dtype=torch.float32)

    def _transition_score(src_label: int, dst_label: int) -> torch.Tensor:
        if transition_scores is None:
            return zero
        src_index = src_label - _LABEL_OFFSET
        dst_index = dst_label - _LABEL_OFFSET
        if src_index < 0 or dst_index < 0:
            raise ValueError("labels must be >= 1 when applying transition scores")
        return transition_scores[src_index, dst_index]

    first_label = extended[0]
    _add_arc(
        sources,
        dests,
        labels,
        scores,
        src=start_state,
        dest=1,
        label=first_label,
        score=zero,
    )

    # Standard CTC permits the very first frame to emit the first base directly,
    # so provide a parallel start transition that skips the leading blank.
    first_base_index = next(
        (idx for idx, lbl in enumerate(extended) if lbl != BLANK_LABEL), None
    )
    if first_base_index is not None and first_base_index != 0:
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=start_state,
            dest=first_base_index + 1,
            label=extended[first_base_index],
            score=zero,
        )

    for idx, symbol in enumerate(extended):
        state = idx + 1
        label = symbol
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=state,
            dest=state,
            label=label,
            score=_transition_score(label, label),
        )

        if idx + 1 < num_symbols:
            next_label = extended[idx + 1]
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=state,
                dest=idx + 2,
                label=next_label,
                score=_transition_score(label, next_label),
            )

        if (
            label != BLANK_LABEL
            and idx + 2 < num_symbols
            and extended[idx + 2] != label
        ):
            skip_label = extended[idx + 2]
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=state,
                dest=idx + 3,
                label=skip_label,
                score=_transition_score(label, skip_label),
            )

        if (
            label != BLANK_LABEL
            and idx + 1 == num_symbols - 1
            and extended[-1] == BLANK_LABEL
        ):
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=state,
                dest=final_state,
                label=-1,
                score=zero,
            )

        if idx == num_symbols - 1:
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=state,
                dest=final_state,
                label=-1,
                score=zero,
            )

    arcs_tensor = torch.stack(
        [
            torch.tensor(sources, dtype=torch.int32),
            torch.tensor(dests, dtype=torch.int32),
            torch.tensor(labels, dtype=torch.int32),
            torch.zeros(len(sources), dtype=torch.int32),
        ],
        dim=1,
    )

    sort_key = arcs_tensor[:, 0] * (final_state + 1) + arcs_tensor[:, 1]
    order = torch.argsort(sort_key)
    arcs_sorted = arcs_tensor.index_select(0, order)

    score_tensor = torch.stack(scores, dim=0).to(dtype=torch.float32)
    score_tensor = score_tensor.index_select(0, order.to(device=score_tensor.device))

    fsa = k2.Fsa(arcs_sorted)
    fsa.scores = score_tensor.to(device=torch.device("cpu"))

    if target.is_cuda:
        fsa = fsa.to(target.device)
        fsa.scores = score_tensor.to(target.device)
    return k2.arc_sort(fsa)


def build_target_graph(
    target: torch.Tensor,
    transition: torch.Tensor,
    bos: torch.Tensor,
    eos: torch.Tensor,
) -> "k2.Fsa":
    """Create the numerator graph for a single reference sequence."""

    k2 = _require_k2()
    if target.dim() != 1:
        raise ValueError("target must be a 1D tensor of base indices")

    if target.numel() == 0:
        raise ValueError("target sequence must be non-empty")

    if torch.any(target <= 0):
        raise ValueError("target sequence must contain 1-based base indices without blanks")

    enter_indices = (target.long() - 1).clamp(min=0, max=NUM_BASES - 1)
    extend_indices = enter_indices + STATE_EXTEND_OFFSET

    start_state = 0
    num_positions = target.numel()
    final_state = 1 + num_positions * 2

    sources: List[int] = []
    dests: List[int] = []
    labels: List[int] = []
    scores: List[torch.Tensor] = []

    def enter_node(pos: int) -> int:
        return 1 + pos * 2

    def extend_node(pos: int) -> int:
        return 1 + pos * 2 + 1

    # Start transitions must first emit an ``enter`` to ensure the first
    # character is produced before any ``extend`` state is visited. This keeps
    # the runtime semantics aligned with evaluation, where only ``enter``
    # transitions emit characters.
    first_enter = enter_indices[0]
    _add_arc(
        sources,
        dests,
        labels,
        scores,
        src=start_state,
        dest=enter_node(0),
        label=int(first_enter.item()) + _LABEL_OFFSET,
        score=bos[first_enter],
    )

    for pos in range(num_positions):
        enter_state = int(enter_indices[pos].item())
        extend_state = int(extend_indices[pos].item())
        enter_lbl = enter_state + _LABEL_OFFSET
        extend_lbl = extend_state + _LABEL_OFFSET

        # enter -> extend within the same position
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=enter_node(pos),
            dest=extend_node(pos),
            label=extend_lbl,
            score=transition[enter_state, extend_state],
        )

        # extend -> extend self loop
        _add_arc(
            sources,
            dests,
            labels,
            scores,
            src=extend_node(pos),
            dest=extend_node(pos),
            label=extend_lbl,
            score=transition[extend_state, extend_state],
        )

        if pos + 1 < num_positions:
            next_enter_state = int(enter_indices[pos + 1].item())
            next_enter_lbl = next_enter_state + _LABEL_OFFSET
            prev_extend_state = extend_state
            prev_enter_state = enter_state

            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=extend_node(pos),
                dest=enter_node(pos + 1),
                label=next_enter_lbl,
                score=transition[prev_extend_state, next_enter_state],
            )
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=enter_node(pos),
                dest=enter_node(pos + 1),
                label=next_enter_lbl,
                score=transition[prev_enter_state, next_enter_state],
            )
        else:
            # Final arcs from the last position.
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=enter_node(pos),
                dest=final_state,
                label=-1,
                score=eos[enter_state],
            )
            _add_arc(
                sources,
                dests,
                labels,
                scores,
                src=extend_node(pos),
                dest=final_state,
                label=-1,
                score=eos[extend_state],
            )

    if not sources:
        raise RuntimeError("target graph contains no finite arcs")

    arcs_tensor = torch.stack(
        [
            torch.tensor(sources, dtype=torch.int32),
            torch.tensor(dests, dtype=torch.int32),
            torch.tensor(labels, dtype=torch.int32),
            torch.zeros(len(sources), dtype=torch.int32),
        ],
        dim=1,
    )
    final_state = final_state  # for clarity
    sort_key = arcs_tensor[:, 0] * (final_state + 1) + arcs_tensor[:, 1]
    order = torch.argsort(sort_key)
    arcs_sorted = arcs_tensor.index_select(0, order)
    score_tensor = torch.stack(scores, dim=0).to(dtype=torch.float32, device=transition.device)
    order_device = order.to(score_tensor.device)
    score_tensor = score_tensor.index_select(0, order_device)

    k2 = _require_k2()
    fsa = k2.Fsa(arcs_sorted)
    fsa.scores = score_tensor.to(device=torch.device("cpu"))

    if transition.is_cuda:
        fsa = fsa.to(transition.device)
        fsa.scores = score_tensor
    return k2.arc_sort(fsa)


def create_fsa_vec(graphs: Sequence["k2.Fsa"]) -> "k2.Fsa":
    """Concatenate a sequence of FSAs into a single FsaVec."""

    if not graphs:
        raise ValueError("Expected at least one FSA to create a vector")
    vec_graphs = list(graphs)
    if not vec_graphs:
        raise ValueError("Expected at least one FSA to create a vector")
    first_device = vec_graphs[0].device
    k2_utils = _require_k2_utils()
    cpu_graphs = []
    for graph in vec_graphs:
        if graph.device != first_device:
            raise ValueError("All FSAs must reside on the same device")
        cpu_graphs.append(graph.to("cpu") if first_device.type != "cpu" else graph)
    fsa_vec = k2_utils.create_fsa_vec(cpu_graphs)
    if first_device.type != "cpu":
        fsa_vec = fsa_vec.to(first_device)
        fsa_vec.scores = fsa_vec.scores.to(first_device)
    return fsa_vec


__all__ = [
    "DenseFsaInputs",
    "build_dense_inputs",
    "build_transition_graph",
    "build_target_graph",
    "build_blank_dense_inputs",
    "build_blank_denominator_graph",
    "build_ctc_blank_target_graph",
    "create_fsa_vec",
    "NUM_BASES",
    "NUM_STATES",
    "STATE_EXTEND_OFFSET",
    "BLANK_LABEL",
    "NUM_BLANK_CLASSES",
]