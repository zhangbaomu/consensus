"""Utility helpers for NanoGraph PathFinder2.

The module lazily exposes commonly used helpers to avoid importing optional
dependencies (such as PyTorch) when they are not required.  This is useful for
lightweight utilities like the dataset validation script which only relies on
``ngpathfinder.utils`` to access ``data_validation``.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["load_checkpoint", "save_checkpoint", "configure_logging", "section"]


def __getattr__(name: str) -> Any:
    if name in {"load_checkpoint", "save_checkpoint"}:
        module = importlib.import_module(".checkpoint", __name__)
    elif name == "configure_logging":
        module = importlib.import_module(".logging", __name__)
    elif name == "section":
        module = importlib.import_module(".profiler", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value

