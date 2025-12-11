"""Loss utilities exports."""
from __future__ import annotations

from .base import LOSS_REGISTRY, LossWrapper
from .duration import DurationBCELoss
from .cat_ctc_crf_blank import CatCTCCRFBlankLoss
from .temporal_order import TemporalOrderLoss
from .rnnt import RNNTLoss

try:  # pragma: no cover - optional dependency guard
    from .ctc_crf import CTCCRFNegativeLogLikelihood
except ImportError:  # pragma: no cover
    CTCCRFNegativeLogLikelihood = None  # type: ignore[assignment]

from .ctc_fast import CTCFastLoss

__all__ = [
    "LOSS_REGISTRY",
    "LossWrapper",
    "CTCFastLoss",
    "CatCTCCRFBlankLoss",
    "DurationBCELoss",
    "TemporalOrderLoss",
    "RNNTLoss",
]

if CTCCRFNegativeLogLikelihood is not None:
    __all__.append("CTCCRFNegativeLogLikelihood")