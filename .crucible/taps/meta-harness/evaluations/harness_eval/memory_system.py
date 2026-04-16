"""Base interface for meta-harness ``MemorySystem`` candidates.

Candidates either subclass ``MemorySystem`` directly or provide a class
with the same method names (duck typing). Crucible's
:mod:`crucible.researcher.candidate_validation` checks for both forms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemorySystem(ABC):
    """Abstract interface for a harness candidate.

    Modeled after Stanford IRIS Lab's meta-harness. Candidates are
    self-contained Python files so they can be stored, diffed, and
    re-proposed by the LLM proposer loop without relying on shared
    module state.
    """

    @abstractmethod
    def predict(self, example: dict[str, Any]) -> Any:
        """Return a prediction for a single example dict."""

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Optional: update internal state from a batch of (prediction, feedback).

        Default implementation is a no-op so stateless candidates don't have
        to override it. Stateful candidates (few-shot retrieval, etc.)
        override this to accumulate examples.
        """

    def get_state(self) -> dict[str, Any]:
        """Optional: serialize internal state for checkpointing."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Optional: restore internal state from a serialized snapshot."""
