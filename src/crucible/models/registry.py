from __future__ import annotations

from typing import Any, Callable, Type

_REGISTRY: dict[str, Callable[..., Any]] = {}
_FAMILY_SCHEMAS: dict[str, dict] = {}


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


def register_schema(name: str, schema: dict) -> None:
    """Register a parameter schema for a model family.

    *schema* maps parameter names to dicts of ``{type, default, description}``.
    """
    _FAMILY_SCHEMAS[name] = schema


def get_family_schema(name: str) -> dict:
    """Return the parameter schema for *name*, or a generic stub."""
    return _FAMILY_SCHEMAS.get(name, {"note": "No schema registered for this family"})


def list_families(*, include_builtins: bool = True) -> list[str]:
    """Return the sorted list of registered model family names.

    When *include_builtins* is True (default) and torch-based architectures
    haven't been imported yet, attempts to import them.  Falls back to the
    known built-in names if torch is unavailable.
    """
    if not _REGISTRY and include_builtins:
        try:
            import crucible.models.architectures  # noqa: F401 — triggers registration
        except ImportError:
            # torch not installed — return known built-in names
            return sorted([
                "baseline", "looped", "convloop", "memory", "prefix_memory",
            ])
    return sorted(_REGISTRY)
