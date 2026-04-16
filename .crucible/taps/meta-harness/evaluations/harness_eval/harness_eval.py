"""Standalone multi-metric harness evaluator.

Designed to be importable locally (tests, dry-runs) and from the
``harness_runner`` launcher on pods. Keeps the measurement contract in
one place so every candidate reports the same metrics: accuracy,
tokens_per_example, latency_ms, and a legacy ``val_bpb`` proxy.
"""
from __future__ import annotations

import time
from typing import Any, Callable

try:  # The eval plugin ships its own base class; optional for duck-typed candidates.
    from memory_system import MemorySystem  # type: ignore
except Exception:  # pragma: no cover - tolerate missing file in dev environments
    MemorySystem = object  # type: ignore


def evaluate_harness(
    harness: "MemorySystem",
    dataset: list[dict[str, Any]],
    *,
    epochs: int = 1,
    score_fn: Callable[[Any, Any], float] | None = None,
) -> dict[str, float]:
    """Run a candidate through the dataset for ``epochs`` epochs.

    Parameters
    ----------
    harness:
        Instance of a ``MemorySystem``-compatible class.
    dataset:
        List of example dicts; each should carry a ``target`` key.
    score_fn:
        Optional ``(prediction, target) -> score`` mapping; defaults to
        equality with basic coercion for dict-wrapped predictions.

    Returns
    -------
    Dict of metrics: ``accuracy``, ``tokens_per_example``, ``latency_ms``,
    plus a ``val_bpb`` proxy (1 - accuracy) so legacy Crucible tooling
    still sees a primary metric.
    """
    def _default_score(pred: Any, target: Any) -> float:
        if isinstance(pred, dict):
            pred = pred.get("label", pred.get("prediction"))
        return 1.0 if pred == target else 0.0

    scorer = score_fn or _default_score
    total_correct = 0.0
    total_tokens = 0
    total_latency = 0.0
    total_examples = 0

    for _ in range(max(epochs, 1)):
        batch: list[dict[str, Any]] = []
        for example in dataset:
            start = time.time()
            pred = harness.predict(example)
            total_latency += (time.time() - start) * 1000.0
            target = example.get("target")
            total_correct += scorer(pred, target)
            total_tokens += len(str(pred))
            total_examples += 1
            batch.append({"example": example, "prediction": pred})
        if hasattr(harness, "learn_from_batch"):
            harness.learn_from_batch(batch)

    n = max(total_examples, 1)
    accuracy = total_correct / n
    return {
        "accuracy": accuracy,
        "tokens_per_example": total_tokens / n,
        "latency_ms": total_latency / n,
        "val_bpb": max(1e-6, 1.0 - accuracy),
    }
