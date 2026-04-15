"""Pluggable training callback registry.

Callbacks hook into the training loop lifecycle (``on_train_begin``,
``on_step_end``, ``on_validation_end``, ``on_train_end``).  They are
registered by name and selected via the ``CALLBACKS`` env var
(comma-separated).

Usage::

    from crucible.training.callbacks import build_callbacks
    cbs = build_callbacks("grad_clip,nan_detector", max_grad_norm=1.0)
    for cb in cbs:
        cb.on_step_end(step, metrics, state)
"""
from __future__ import annotations

from abc import ABC
from typing import Any

from crucible.core.plugin_registry import PluginRegistry

CALLBACK_REGISTRY = PluginRegistry("callback")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TrainingCallback(ABC):
    """Lifecycle hook for the training loop.

    All methods are optional — subclasses override only what they need.
    ``priority`` controls execution order (lower = runs first).
    """

    priority: int = 100

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Called after model creation but BEFORE torch.compile.

        This is the correct hook for registering forward pre-hooks
        (e.g. QAT fake quantization, activation collection) that must
        be visible to the compiled graph.
        """

    def on_train_begin(self, state: dict[str, Any]) -> None:
        """Called once before the training loop starts (AFTER torch.compile)."""

    def on_step_begin(self, step: int, state: dict[str, Any]) -> None:
        """Called at the start of each training step."""

    def on_after_backward(self, step: int, state: dict[str, Any]) -> None:
        """Called after loss.backward() but BEFORE optimizer.step().

        This is the correct hook for gradient clipping/manipulation.
        """

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        """Called after each training step (forward + backward + optimize)."""

    def on_validation_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        """Called after a validation pass."""

    def on_train_end(self, state: dict[str, Any]) -> None:
        """Called once after the training loop finishes."""


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------

class GradClipCallback(TrainingCallback):
    """Gradient norm clipping — fires after backward, before optimizer step."""

    priority = 10  # Run early

    def __init__(self, *, max_grad_norm: float = 1.0, **kwargs: Any) -> None:
        self.max_grad_norm = max_grad_norm

    def on_after_backward(self, step: int, state: dict[str, Any]) -> None:
        model = state.get("model")
        if model is None:
            return
        import torch
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)


class NaNDetectorCallback(TrainingCallback):
    """Detect NaN/Inf in training loss and raise."""

    priority = 20

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        import math
        loss = metrics.get("train_loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            raise RuntimeError(f"NaN/Inf detected at step {step}: train_loss={loss}")


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when a metric hasn't improved for N steps."""

    priority = 90

    def __init__(self, *, patience: int = 500, metric: str = "val_loss", mode: str = "min", **kwargs: Any) -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self._best: float | None = None
        self._wait = 0

    def on_validation_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        val = metrics.get(self.metric)
        if val is None:
            return
        if self._best is None:
            self._best = val
            return
        improved = val < self._best if self.mode == "min" else val > self._best
        if improved:
            self._best = val
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                state["should_stop"] = True


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def register_callback(name: str, factory: type[TrainingCallback], *, source: str = "builtin") -> None:
    CALLBACK_REGISTRY.register(name, factory, source=source)


def build_callback(name: str, **kwargs: Any) -> TrainingCallback:
    """Build a single callback by name."""
    factory = CALLBACK_REGISTRY.get(name)
    if factory is None:
        from crucible.core.errors import PluginError
        available = ", ".join(CALLBACK_REGISTRY.list_plugins()) or "(none)"
        raise PluginError(f"Unknown callback {name!r}. Registered: {available}")
    return factory(**kwargs)


def build_callbacks(names: str, **kwargs: Any) -> list[TrainingCallback]:
    """Build callbacks from a comma-separated string, sorted by priority."""
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    cbs = [build_callback(n, **kwargs) for n in name_list]
    cbs.sort(key=lambda cb: cb.priority)
    return cbs


def list_callbacks() -> list[str]:
    return CALLBACK_REGISTRY.list_plugins()


def list_callbacks_detailed() -> list[dict[str, str]]:
    return CALLBACK_REGISTRY.list_plugins_detailed()


# ---------------------------------------------------------------------------
# Register built-ins
# ---------------------------------------------------------------------------

register_callback("grad_clip", GradClipCallback)
register_callback("nan_detector", NaNDetectorCallback)
register_callback("early_stopping", EarlyStoppingCallback)
