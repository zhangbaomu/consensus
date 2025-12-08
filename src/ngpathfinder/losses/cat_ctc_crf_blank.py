"""CAT-backed CTC-CRF loss variant tailored for blank-aware decoders."""

from __future__ import annotations

import hashlib
import importlib
import importlib.resources as importlib_resources
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .base import LOSS_REGISTRY
from .ctc_common import _gather_reference, minimal_ctc_input_length


_TRUTHY_ENV = {"1", "true", "yes", "on", "enable", "enabled"}


def _env_flag(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in _TRUTHY_ENV


class _LazyModuleLoader:
    """Utility providing late binding of the optional ``ctc_crf`` backend."""

    _FALLBACKS = {
        "ctc_crf": ("cat.ctc_crf", "cat.ctc_crf.ctc_crf"),
        "cat.ctc_crf": ("ctc_crf",),
    }

    def __init__(self, module_name: str) -> None:
        self._preferred_name = module_name
        self._resolved_name: Optional[str] = None
        self._module: Optional[object] = None

    def _candidate_names(self) -> List[str]:
        seen: set[str] = set()
        candidates: List[str] = []

        env_override = os.environ.get("NGPATHFINDER_CAT_BACKEND")
        if env_override:
            for entry in env_override.split(","):
                candidate = entry.strip()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

        if self._preferred_name and self._preferred_name not in seen:
            seen.add(self._preferred_name)
            candidates.append(self._preferred_name)

        for fallback in self._FALLBACKS.get(self._preferred_name, ()):  # pragma: no branch - finite enumeration
            if fallback not in seen:
                seen.add(fallback)
                candidates.append(fallback)

        return candidates

    def load(self) -> object:
        if self._module is None:
            last_error: Optional[Exception] = None
            candidates = self._candidate_names()
            for name in candidates:
                try:
                    module = importlib.import_module(name)
                except ModuleNotFoundError as exc:
                    last_error = exc
                    continue
                self._module = module
                self._resolved_name = module.__name__
                break

            if self._module is None:
                raise ModuleNotFoundError(
                    "Unable to import CAT backend. Tried candidates: %s" % (", ".join(candidates) or self._preferred_name)
                ) from last_error

        return self._module

    @property
    def resolved_name(self) -> Optional[str]:
        return self._resolved_name


def _flatten_targets(targets: Sequence[Tensor]) -> Tensor:
    if not targets:
        raise ValueError("Targets must contain at least one reference sequence")
    return torch.cat([t.to(dtype=torch.long) for t in targets], dim=0)


def _cache_root() -> Path:
    cache_env = os.environ.get("NGPATHFINDER_CACHE_HOME")
    if cache_env:
        return Path(cache_env)
    return Path(tempfile.gettempdir()) / "ngpathfinder"


def _materialize_resource(package: str, resource: str) -> Optional[str]:
    try:
        package_module = importlib.import_module(package)
    except ModuleNotFoundError:
        return None

    data: Optional[bytes] = None

    try:
        files = importlib_resources.files(package_module)
    except (AttributeError, TypeError):
        files = None

    if files is not None:
        traversable = files.joinpath(resource)
        if traversable.is_file():
            data = traversable.read_bytes()

    if data is None:
        fallback_loader = getattr(package_module, "denominator_wfst_bytes", None)
        if callable(fallback_loader):
            data = fallback_loader()

    if data is None:
        return None
    digest = hashlib.sha1(data).hexdigest()
    suffix = Path(resource).suffix or ".fst"
    cache_dir = _cache_root() / "cat"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"den_lm_{digest}{suffix}"
    if not dest.exists():
        dest.write_bytes(data)
    return str(dest)


def _resolve_denominator_path(
    den_lm_path: Optional[str],
    backend_module: str,
) -> str:
    """Resolve the denominator WFST path for CAT.

    The resolver supports several forms:

    - Absolute or relative filesystem paths.
    - ``pkg://`` URIs pointing to resources (e.g. ``pkg://ctc_crf::data/den_lm.fst``).
    - ``auto``/``default`` sentinels that search the backend module for a packaged
      ``den_lm.fst`` resource.

    ``NGPATHFINDER_CAT_DEN_LM`` overrides ``den_lm_path`` when the value is omitted or
    set to an automatic sentinel.
    """

    sentinel_values = {None, "", "auto", "default"}

    env_override = os.environ.get("NGPATHFINDER_CAT_DEN_LM")
    if den_lm_path in sentinel_values and env_override:
        den_lm_path = env_override

    if den_lm_path not in sentinel_values:
        expanded = os.path.expandvars(os.path.expanduser(str(den_lm_path)))
        if expanded.startswith("pkg://"):
            spec = expanded[len("pkg://") :]
            if "::" in spec:
                pkg, resource = spec.split("::", 1)
            else:
                pkg, resource = spec, "den_lm.fst"
            resolved = _materialize_resource(pkg, resource)
            if resolved:
                return resolved
            raise FileNotFoundError(
                f"Failed to locate packaged denominator WFST '{resource}' in '{pkg}'"
            )

        if "::" in expanded:
            pkg, resource = expanded.split("::", 1)
            resolved = _materialize_resource(pkg, resource)
            if resolved:
                return resolved
            raise FileNotFoundError(
                f"Failed to locate packaged denominator WFST '{resource}' in '{pkg}'"
            )

        path = Path(expanded)
        if path.is_file():
            return str(path)
        project_relative = Path.cwd() / path
        if project_relative.is_file():
            return str(project_relative)

        resolved = _materialize_resource(backend_module, expanded)
        if resolved:
            return resolved

        raise FileNotFoundError(f"Denominator WFST not found: {den_lm_path}")

    # Auto-detection path: search common resource locations packaged with backend.
    resource_candidates = [
        "den_lm.fst",
        "data/den_lm.fst",
        "resources/den_lm.fst",
        "assets/den_lm.fst",
    ]
    for candidate in resource_candidates:
        resolved = _materialize_resource(backend_module, candidate)
        if resolved:
            return resolved

    packaged_fallbacks = [
        ("ngpathfinder.resources.cat", "den_lm.fst"),
    ]
    for package, resource in packaged_fallbacks:
        resolved = _materialize_resource(package, resource)
        if resolved:
            return resolved

    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    search_roots.extend(Path(__file__).resolve().parents)
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / "signal_data" / "cat" / "den_lm.fst"
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Could not auto-detect a denominator WFST. Provide 'den_lm_path' or package "
        "one under the CAT backend module."
    )


@LOSS_REGISTRY.register("cat_ctc_crf_blank")
class CatCTCCRFBlankLoss(nn.Module):
    """Compute the CTC-CRF objective using the CAT CUDA extension.

    The module expects decoder outputs that follow the ``ctc_crf_blank`` layout used by
    :class:`~src.ngpathfinder.losses.ctc_crf.CTCCRFNegativeLogLikelihood`, namely a
    ``ctc_blank_logits`` tensor containing ``{blank, A, C, G, T}`` scores. Targets are
    pulled from ``batch['reference_index']`` in the same way as the k2 variant.

    Parameters
    ----------
    den_lm_path:
        File path to the denominator WFST compiled for the CAT toolkit.
    lamb:
        Auxiliary weight applied to the numerator CTC term (``lambda`` in CAT docs).
    backend_module:
        Name of the Python module exposing ``CTC_CRF_LOSS`` and ``CRFContext``.
        ``NGPATHFINDER_CAT_BACKEND`` overrides this at runtime and accepts a
        comma-separated list of module candidates.
    apply_log_softmax:
        Whether to apply ``log_softmax`` to logits before dispatching to CAT.
    enforce_length_check:
        If ``True`` (default) the loss will verify that every sequence length meets the
        minimal requirement imposed by CTC collapse rules and raise a descriptive
        :class:`RuntimeError` otherwise.
    allow_cpu_fallback:
        CAT only supports CUDA execution. This flag exists solely to aid unit testing
        with mocked backends; enabling it in production will raise at runtime if the
        real backend is used on CPU. Set ``NGPATHFINDER_CAT_ALLOW_CPU_FALLBACK`` to
        override the runtime behavior without editing configuration files.
    """

    def __init__(
        self,
        *,
        den_lm_path: str | None = None,
        lamb: float = 0.1,
        backend_module: str = "ctc_crf",
        apply_log_softmax: bool = True,
        enforce_length_check: bool = True,
        allow_cpu_fallback: bool = False,
        logit_key: str = "ctc_blank_logits",
        length_key: str = "decoder_lengths",
        padding_mask_key: str = "decoder_padding_mask",
    ) -> None:
        super().__init__()

        backend_name = backend_module or "ctc_crf"
        if backend_name in {"", "auto", "default"}:
            backend_name = "ctc_crf"

        self._module_loader = _LazyModuleLoader(backend_name)
        module = self._module_loader.load()

        try:
            loss_builder = getattr(module, "CTC_CRF_LOSS")
            context_builder = getattr(module, "CRFContext")
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                f"Module '{backend_module}' does not expose CAT CTC-CRF bindings"
            ) from exc

        resolved_den_path = _resolve_denominator_path(den_lm_path, module.__name__)

        criterion = loss_builder(lamb=lamb)
        if not callable(criterion):  # pragma: no cover - sanity guard
            raise TypeError(
                "CTC_CRF_LOSS must return a callable that computes the CAT loss"
            )

        env_cpu_flag = _env_flag("NGPATHFINDER_CAT_ALLOW_CPU_FALLBACK")
        if env_cpu_flag is not None:
            allow_cpu_fallback = env_cpu_flag

        self._criterion = criterion
        self._context_builder = context_builder
        self._den_lm_path = resolved_den_path
        self._apply_log_softmax = bool(apply_log_softmax)
        self._enforce_length_check = bool(enforce_length_check)
        self._allow_cpu_fallback = bool(allow_cpu_fallback)

        self._logit_key = logit_key
        self._length_key = length_key
        self._padding_mask_key = padding_mask_key

        self._crf_context = None
        self._crf_device_index: int | None = None
        self._debug_calls = 0

    def _ensure_context(self, device: torch.device) -> None:
        if device.type != "cuda":
            if self._allow_cpu_fallback:
                return
            raise RuntimeError(
                "CAT CTC-CRF requires CUDA tensors; received device '%s'" % device
            )

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        if self._crf_context is not None and self._crf_device_index == device_index:
            return

        self._crf_context = self._context_builder(self._den_lm_path, device_index)
        self._crf_device_index = device_index

    def _extract_lengths(
        self, logits: Tensor, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        batch_size, time_steps, _ = logits.shape

        lengths = outputs.get(self._length_key)
        if lengths is None:
            lengths = torch.full(
                (batch_size,),
                time_steps,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            lengths = lengths.to(device=logits.device, dtype=torch.long)

        pad_mask = outputs.get(self._padding_mask_key)
        if pad_mask is None:
            pad_mask = batch.get(self._padding_mask_key)

        if pad_mask is not None:
            if pad_mask.shape != logits.shape[:2]:
                raise ValueError(
                    "decoder_padding_mask must match logits shape (batch, time)"
                )
            pad_bool = pad_mask.to(dtype=torch.bool, device=logits.device)
            lengths = (~pad_bool).sum(dim=1).to(dtype=torch.long)
            logits = logits.masked_fill(pad_bool.unsqueeze(-1), float("-inf"))

        return lengths, logits

    def _check_lengths(
        self,
        targets: Sequence[Tensor],
        target_lengths: Iterable[int],
        frame_lengths: Tensor,
        segment_ids: Sequence[str],
    ) -> None:
        if not self._enforce_length_check:
            return

        insufficient: List[str] = []
        for idx, (target, tgt_len) in enumerate(zip(targets, target_lengths)):
            required = minimal_ctc_input_length(target, tgt_len)
            available = int(frame_lengths[idx].item())
            if required > available:
                segment = segment_ids[idx] if idx < len(segment_ids) else str(idx)
                insufficient.append(
                    f"{segment}:required={required}>available={available}"
                )
        if insufficient:
            raise RuntimeError(
                "CTC-CRF blank variant detected insufficient decoder frames for CAT: "
                + ", ".join(insufficient)
            )

    def forward(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tensor:
        logits = outputs.get(self._logit_key)
        if logits is None:
            raise KeyError(
                f"Decoder outputs must contain '{self._logit_key}' for CAT CTC-CRF"
            )
        if logits.dim() != 3:
            raise ValueError("ctc_blank_logits must have shape (batch, time, classes)")

        original_dtype = logits.dtype
        original_device = logits.device

        if self._apply_log_softmax:
            logits = torch.log_softmax(logits, dim=-1)
        else:
            logits = logits.clone()

        logits = logits.to(dtype=torch.float32)
        if not logits.is_contiguous():
            logits = logits.contiguous()
        lengths, logits = self._extract_lengths(logits, outputs, batch)

        raw_targets, reference_lengths = _gather_reference(batch)
        targets = [t.to(dtype=torch.long) for t in raw_targets]
        target_lengths = reference_lengths.to(dtype=torch.long)

        if torch.any(target_lengths <= 0):
            raise ValueError("CAT CTC-CRF requires strictly positive target lengths")

        segment_ids = batch.get("segment_id")
        if isinstance(segment_ids, list):
            segment_list = [str(seg) for seg in segment_ids]
        elif torch.is_tensor(segment_ids):
            segment_list = [str(item) for item in segment_ids.view(-1).tolist()]
        elif segment_ids is None:
            segment_list = []
        else:
            segment_list = [str(segment_ids)]

        self._check_lengths(targets, target_lengths.tolist(), lengths, segment_list)

        flat_targets = _flatten_targets(targets).contiguous()
        flat_targets_cpu = flat_targets.to(
            device=torch.device("cpu"), dtype=torch.int32
        )
        lengths_cpu = lengths.to(device=torch.device("cpu"), dtype=torch.int32)
        target_lengths_cpu = target_lengths.to(
            device=torch.device("cpu"), dtype=torch.int32
        )

        self._ensure_context(logits.device)

        loss = self._call_backend(
            logits,
            flat_targets_cpu,
            lengths_cpu,
            target_lengths_cpu,
        )

        self._debug_calls += 1
        logger = logging.getLogger("cat_ctc_crf")
        should_report = self._debug_calls <= 5
        if should_report:
            logger.info(
                "cat_ctc_crf blank batch | segments=%s | loss=%.6f",
                segment_list if segment_list else batch.get("segment_id", []),
                float(loss.detach().cpu().item()),
            )

        return loss.to(dtype=original_dtype, device=original_device)

    def _call_backend(
        self,
        logits: Tensor,
        flat_targets: Tensor,
        frame_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Dispatch to the CAT CUDA extension accommodating API variations."""

        context: Optional[object] = self._crf_context
        args: Tuple[Tensor, Tensor, Tensor, Tensor]
        args = (logits, flat_targets, frame_lengths, target_lengths)

        if context is not None:
            try:
                return self._criterion(*args, context)
            except TypeError:
                # Some builds expose keyword-only context parameters.
                try:
                    return self._criterion(*args, context=context)
                except TypeError:
                    pass

        try:
            return self._criterion(*args)
        except TypeError as exc:  # pragma: no cover - surfaced when context is mandatory
            raise TypeError(
                "CAT CTC-CRF backend rejected provided arguments; ensure your CAT"
                " installation exposes CTC_CRF_LOSS(logits, targets, frame_lengths,"
                " target_lengths[, context])"
            ) from exc


__all__ = ["CatCTCCRFBlankLoss"]