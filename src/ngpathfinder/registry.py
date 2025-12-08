"""Lightweight registry utilities for pluggable modules."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, MutableMapping

Registry = MutableMapping[str, Callable[..., Any]]


class ModuleRegistry:
    """Simple string-to-constructor registry."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if name in self._registry:
            raise KeyError(f"Registry already contains a component named '{name}'")

        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            self._registry[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"Unknown component '{name}'. Available: {list(self._registry)}") from exc

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        factory = self.get(name)
        return factory(*args, **kwargs)

    def available(self) -> Registry:
        return dict(self._registry)


registries: Dict[str, ModuleRegistry] = defaultdict(ModuleRegistry)


def get_registry(namespace: str) -> ModuleRegistry:
    return registries[namespace]


__all__ = ["ModuleRegistry", "get_registry"]
