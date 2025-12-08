"""Model component namespaces."""
from .encoder import ENCODER_REGISTRY
from .fusion import FUSION_REGISTRY
from .aggregator import AGGREGATOR_REGISTRY
from .decoder import DECODER_REGISTRY

__all__ = [
    "ENCODER_REGISTRY",
    "FUSION_REGISTRY",
    "AGGREGATOR_REGISTRY",
    "DECODER_REGISTRY",
]
