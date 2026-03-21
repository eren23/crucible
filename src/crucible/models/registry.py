from __future__ import annotations

from typing import Any, Callable, Type

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str, factory: Callable[..., Any]) -> None:
    """Register a model family under *name*.

    *factory* is a callable ``(args) -> nn.Module`` that accepts an
    argparse-style namespace and returns a constructed model.
    """
    if name in _REGISTRY:
        raise ValueError(f"Model family {name!r} is already registered")
    _REGISTRY[name] = factory


def build_model(args: Any) -> Any:
    """Build a model from an argparse-style namespace.

    ``args.model_family`` selects the registered factory.
    """
    family = getattr(args, "model_family", None)
    if family is None:
        raise ValueError("args must have a 'model_family' attribute")
    if family not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"unsupported MODEL_FAMILY={family!r}; registered families: {available}"
        )
    return _REGISTRY[family](args)


def list_families() -> list[str]:
    """Return the sorted list of registered model family names."""
    return sorted(_REGISTRY)
