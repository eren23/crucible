"""Baseline: rolling few-shot memory with majority vote over seen examples."""
from __future__ import annotations

from collections import Counter
from typing import Any


class MemorySystem:
    """Maintains a fixed-size example window and votes on the majority label."""

    def __init__(self, window_size: int = 16) -> None:
        self.window_size = window_size
        self._examples: list[dict[str, Any]] = []

    def predict(self, example: dict[str, Any]) -> int:
        if not self._examples:
            return 0
        counts = Counter(e.get("target") for e in self._examples)
        return int(counts.most_common(1)[0][0])

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = r.get("example") or {}
            if "target" in ex:
                self._examples.append(ex)
                if len(self._examples) > self.window_size:
                    self._examples.pop(0)

    def get_state(self) -> dict[str, Any]:
        return {"examples": list(self._examples), "window_size": self.window_size}

    def set_state(self, state: dict[str, Any]) -> None:
        self._examples = list(state.get("examples") or [])
        self.window_size = int(state.get("window_size", self.window_size))
