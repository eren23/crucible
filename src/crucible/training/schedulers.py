"""Pluggable LR scheduler registry.

Every scheduler is registered as a factory callable that accepts
``(optimizer, *, total_steps, warmup_steps, **kwargs)`` and returns a
scheduler (or ``None`` for no scheduling).

Built-in schedulers are registered at import time.  External plugins
are discovered from ``.crucible/plugins/schedulers/`` and
``~/.crucible-hub/plugins/schedulers/``.

Usage in training backends::

    from crucible.training.schedulers import build_scheduler
    scheduler = build_scheduler("cosine", optimizer, total_steps=10000, warmup_steps=200)
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from crucible.core.plugin_registry import PluginRegistry

SCHEDULER_REGISTRY: PluginRegistry[Callable[..., Any]] = PluginRegistry("scheduler")


# ---------------------------------------------------------------------------
# Convenience wrappers (public API)
# ---------------------------------------------------------------------------

def register_scheduler(name: str, factory: Callable[..., Any], *, source: str = "builtin") -> None:
    """Register a scheduler factory under *name*."""
    SCHEDULER_REGISTRY.register(name, factory, source=source)


def build_scheduler(name: str, optimizer: Any, **kwargs: Any) -> Any:
    """Build a scheduler by name.

    *optimizer* is the torch optimizer.  All remaining *kwargs*
    (``total_steps``, ``warmup_steps``, etc.) are forwarded to the factory.
    Returns ``None`` if the factory decides no scheduling is needed.
    """
    factory = SCHEDULER_REGISTRY.get(name)
    if factory is None:
        from crucible.core.errors import PluginError
        available = ", ".join(SCHEDULER_REGISTRY.list_plugins()) or "(none)"
        raise PluginError(
            f"Unknown scheduler {name!r}. Registered: {available}"
        )
    return factory(optimizer, **kwargs)


def list_schedulers() -> list[str]:
    """Return sorted list of registered scheduler names."""
    return SCHEDULER_REGISTRY.list_plugins()


def list_schedulers_detailed() -> list[dict[str, str]]:
    """Return schedulers with source metadata."""
    return SCHEDULER_REGISTRY.list_plugins_detailed()


# ---------------------------------------------------------------------------
# Built-in scheduler factories
# ---------------------------------------------------------------------------
# Each factory: (optimizer, *, total_steps, warmup_steps, **kwargs) -> scheduler | None
# Torch is lazy-imported to avoid top-level dependency.


def _constant_factory(optimizer: Any, *, total_steps: int = 0, warmup_steps: int = 0, **kwargs: Any) -> Any:
    """Constant LR (with optional warmup)."""
    if warmup_steps <= 0:
        return None

    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _cosine_factory(optimizer: Any, *, total_steps: int = 0, warmup_steps: int = 0, min_lr_scale: float = 0.0, **kwargs: Any) -> Any:
    """Cosine decay with optional linear warmup."""
    import torch

    if total_steps <= 0 and warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _cosine_restarts_factory(optimizer: Any, *, total_steps: int = 0, warmup_steps: int = 0, num_restarts: int = 1, **kwargs: Any) -> Any:
    """Cosine annealing with warm restarts."""
    import torch

    t_0 = max(1, (total_steps - warmup_steps) // max(1, num_restarts))
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0)


def _linear_factory(optimizer: Any, *, total_steps: int = 0, warmup_steps: int = 0, min_lr_scale: float = 0.0, **kwargs: Any) -> Any:
    """Linear decay with optional warmup."""
    import torch

    if total_steps <= 0 and warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        return min_lr_scale + (1.0 - min_lr_scale) * (1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Register built-ins
# ---------------------------------------------------------------------------

register_scheduler("constant", _constant_factory)
register_scheduler("cosine", _cosine_factory)
register_scheduler("cosine_restarts", _cosine_restarts_factory)
register_scheduler("linear", _linear_factory)

# Schemas for MCP introspection
SCHEDULER_REGISTRY.register_schema("constant", {
    "warmup_steps": {"type": "int", "default": 0, "description": "Linear warmup steps (returns None if 0)"},
})
SCHEDULER_REGISTRY.register_schema("cosine", {
    "total_steps": {"type": "int", "required": True, "description": "Total training steps"},
    "warmup_steps": {"type": "int", "default": 0},
    "min_lr_scale": {"type": "float", "default": 0.0, "description": "Minimum LR as fraction of base LR"},
})
SCHEDULER_REGISTRY.register_schema("cosine_restarts", {
    "total_steps": {"type": "int", "required": True},
    "warmup_steps": {"type": "int", "default": 0},
    "num_restarts": {"type": "int", "default": 1, "description": "Number of cosine restarts"},
})
SCHEDULER_REGISTRY.register_schema("linear", {
    "total_steps": {"type": "int", "required": True},
    "warmup_steps": {"type": "int", "default": 0},
    "min_lr_scale": {"type": "float", "default": 0.0},
})
