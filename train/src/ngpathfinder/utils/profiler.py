"""Training profiler stubs."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def section(name: str) -> Iterator[None]:
    print(f"[Profiler] Entering {name}")
    try:
        yield
    finally:
        print(f"[Profiler] Exiting {name}")


__all__ = ["section"]
