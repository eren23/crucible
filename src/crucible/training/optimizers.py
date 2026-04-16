"""Pluggable optimizer registry.

Every optimizer is registered as a factory callable that accepts
``(params, *, lr, **kwargs)`` and returns a ``torch.optim.Optimizer``.

Built-in optimizers are registered at import time.  External plugins
are discovered from ``.crucible/plugins/optimizers/`` and
``~/.crucible-hub/plugins/optimizers/`` via :func:`discover_all_plugins`.

Usage in training backends::

    from crucible.training.optimizers import build_optimizer
    optimizer = build_optimizer("adamw", params, lr=0.001, weight_decay=0.01)
"""
from __future__ import annotations

from typing import Any

from crucible.core.plugin_registry import PluginRegistry
from crucible.core.types import PluginFactory

OPTIMIZER_REGISTRY: PluginRegistry[PluginFactory] = PluginRegistry("optimizer")


# ---------------------------------------------------------------------------
# Convenience wrappers (public API)
# ---------------------------------------------------------------------------

def register_optimizer(name: str, factory: PluginFactory, *, source: str = "builtin") -> None:
    """Register an optimizer factory under *name*."""
    OPTIMIZER_REGISTRY.register(name, factory, source=source)


def build_optimizer(name: str, params: Any, **kwargs: Any) -> Any:
    """Build an optimizer by name.

    *params* is the parameter list/groups (as accepted by ``torch.optim``).
    All remaining *kwargs* are forwarded to the factory.
    """
    factory = OPTIMIZER_REGISTRY.get(name)
    if factory is None:
        from crucible.core.errors import PluginError
        available = ", ".join(OPTIMIZER_REGISTRY.list_plugins()) or "(none)"
        raise PluginError(
            f"Unknown optimizer {name!r}. Registered: {available}"
        )
    return factory(params, **kwargs)


def list_optimizers() -> list[str]:
    """Return sorted list of registered optimizer names."""
    return OPTIMIZER_REGISTRY.list_plugins()


def list_optimizers_detailed() -> list[dict[str, str]]:
    """Return optimizers with source metadata."""
    return OPTIMIZER_REGISTRY.list_plugins_detailed()


# ---------------------------------------------------------------------------
# Built-in optimizer factories
# ---------------------------------------------------------------------------
# Each factory has signature: (params, **kwargs) -> torch.optim.Optimizer
# Torch is lazy-imported to avoid top-level dependency.


def _adam_factory(params: Any, **kwargs: Any) -> Any:
    import torch
    return torch.optim.Adam(params, **kwargs)


def _adamw_factory(params: Any, **kwargs: Any) -> Any:
    import torch
    return torch.optim.AdamW(params, **kwargs)


def _sgd_factory(params: Any, **kwargs: Any) -> Any:
    import torch
    return torch.optim.SGD(params, **kwargs)


def _rmsprop_factory(params: Any, **kwargs: Any) -> Any:
    import torch
    return torch.optim.RMSprop(params, **kwargs)


def _muon_factory(params: Any, **kwargs: Any) -> Any:
    from crucible.training.muon import Muon
    return Muon(params, **kwargs)


# ---------------------------------------------------------------------------
# Register built-ins
# ---------------------------------------------------------------------------

register_optimizer("adam", _adam_factory)
register_optimizer("adamw", _adamw_factory)
register_optimizer("sgd", _sgd_factory)
register_optimizer("rmsprop", _rmsprop_factory)
register_optimizer("muon", _muon_factory)

# Schemas for MCP introspection
OPTIMIZER_REGISTRY.register_schema("adam", {
    "lr": {"type": "float", "default": 0.001, "description": "Learning rate"},
    "betas": {"type": "tuple", "default": "(0.9, 0.999)", "description": "Coefficients for running averages"},
    "eps": {"type": "float", "default": 1e-8, "description": "Numerical stability term"},
    "weight_decay": {"type": "float", "default": 0, "description": "L2 penalty"},
    "fused": {"type": "bool", "default": False, "description": "Use fused kernel (CUDA only)"},
})
OPTIMIZER_REGISTRY.register_schema("adamw", {
    "lr": {"type": "float", "default": 0.001},
    "betas": {"type": "tuple", "default": "(0.9, 0.999)"},
    "eps": {"type": "float", "default": 1e-8},
    "weight_decay": {"type": "float", "default": 0.01, "description": "Decoupled weight decay"},
    "fused": {"type": "bool", "default": False},
})
OPTIMIZER_REGISTRY.register_schema("sgd", {
    "lr": {"type": "float", "required": True},
    "momentum": {"type": "float", "default": 0},
    "weight_decay": {"type": "float", "default": 0},
    "nesterov": {"type": "bool", "default": False},
})
OPTIMIZER_REGISTRY.register_schema("rmsprop", {
    "lr": {"type": "float", "default": 0.01},
    "alpha": {"type": "float", "default": 0.99},
    "eps": {"type": "float", "default": 1e-8},
    "weight_decay": {"type": "float", "default": 0},
    "momentum": {"type": "float", "default": 0},
})
OPTIMIZER_REGISTRY.register_schema("muon", {
    "lr": {"type": "float", "required": True},
    "momentum": {"type": "float", "default": 0.95, "description": "Newton-Schulz momentum"},
    "backend_steps": {"type": "int", "default": 5, "description": "Newton-Schulz iteration steps"},
    "weight_decay": {"type": "float", "default": 0},
})
