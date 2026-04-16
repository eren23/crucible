"""Baseline: stateless classifier. Ignores learning signal."""
from __future__ import annotations

from typing import Any


class MemorySystem:
    """Simple majority-class baseline."""

    def predict(self, example: dict[str, Any]) -> int:
        # Heuristic: odd example ids -> class 1, otherwise class 0.
        text = str(example.get("input", ""))
        return 1 if len(text) % 2 else 0

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        return
