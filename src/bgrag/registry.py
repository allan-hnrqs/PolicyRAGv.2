"""Simple registries for pluggable architecture components."""

from __future__ import annotations

from collections.abc import Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., object]] = {}

    def register(self, key: str, value: Callable[..., object]) -> None:
        if key in self._items:
            raise ValueError(f"{self.name} already has key {key!r}")
        self._items[key] = value

    def get(self, key: str) -> Callable[..., object]:
        if key not in self._items:
            raise KeyError(f"{key!r} is not registered in {self.name}")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items.keys())


chunker_registry = Registry("chunkers")
metadata_enricher_registry = Registry("metadata_enrichers")
source_topology_registry = Registry("source_topologies")
answer_strategy_registry = Registry("answer_strategies")
