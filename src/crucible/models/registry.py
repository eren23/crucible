from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.log import log_warn
from crucible.core.types import PluginFactory

_REGISTRY: dict[str, PluginFactory] = {}
_REGISTRY_META: dict[str, dict] = {}
_CURRENT_REGISTER_SOURCE: str = "local"
_FAMILY_SCHEMAS: dict[str, dict] = {}


def register_model(name: str, factory: PluginFactory, *, source: str = "") -> None:
    """Register a model family under *name*.

    *factory* is a callable ``(args) -> nn.Module`` that accepts an
    argparse-style namespace and returns a constructed model.

    *source* indicates where the registration originates: ``"builtin"``,
    ``"global"`` (hub), or ``"local"`` (project ``.crucible/architectures/``).
    When omitted, defaults to the current module-level source context
    (``_CURRENT_REGISTER_SOURCE``).

    Higher-precedence sources silently override lower ones
    (``builtin`` < ``global`` < ``local``).  Attempting to register with
    equal or lower precedence raises ``ValueError``.
    """
    source = source or _CURRENT_REGISTER_SOURCE
    if name in _REGISTRY:
        existing_source = _REGISTRY_META.get(name, {}).get("source", "builtin")
        precedence = {"builtin": 0, "global": 1, "local": 2}
        if precedence.get(source, 1) <= precedence.get(existing_source, 0):
            raise ValueError(f"Model family {name!r} is already registered (source: {existing_source})")
        # Higher precedence: silently override
    _REGISTRY[name] = factory
    _REGISTRY_META[name] = {"source": source}


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
            # torch not installed — return known built-in names (includes
            # the "memory" alias registered by prefix_memory.py)
            return sorted([
                "baseline", "looped", "convloop", "memory", "prefix_memory",
            ])
    return sorted(_REGISTRY)


def list_families_detailed(*, include_builtins: bool = True) -> list[dict[str, str]]:
    """Return families with source metadata."""
    if not _REGISTRY and include_builtins:
        try:
            import crucible.models.architectures  # noqa: F401 — triggers registration
        except ImportError:
            return [{"name": n, "source": "builtin"} for n in sorted(["baseline", "looped", "convloop", "memory", "prefix_memory"])]
    return sorted(
        [{"name": name, "source": _REGISTRY_META[name]["source"]} for name in _REGISTRY],
        key=lambda d: d["name"],
    )


def load_global_architectures(hub_arch_dir: Path, *, source: str = "global") -> list[str]:
    """Import all .py and .yaml files from an external architectures directory.

    *source* controls the registry source tag (``"global"`` for hub,
    ``"local"`` for project ``.crucible/architectures/``).

    Three discovery paths are scanned at ``hub_arch_dir``:

    1. ``<hub_arch_dir>/*.py`` — top-level single-file plugins (legacy single
       file install path used for moe, partial_rope, looped_aug, etc.).
    2. ``<hub_arch_dir>/*.yaml`` — declarative architecture specs.
    3. ``<hub_arch_dir>/<name>/<name>.py`` — directory bundle install path
       (used by multi-file plugins like code_wm that depend on sibling
       modules like wm_base). When loading a bundle, the package directory
       is prepended to ``sys.path`` briefly so sibling modules resolve via
       normal Python imports even if the primary file uses them.

    ``.py`` files are imported as Python modules (they call ``register_model``
    themselves).  ``.yaml`` files are loaded as declarative architecture specs
    via the composer engine.

    Returns list of successfully loaded file stems (or bundle names).
    """
    import importlib.util
    import sys

    loaded: list[str] = []
    global _CURRENT_REGISTER_SOURCE
    prev_source = _CURRENT_REGISTER_SOURCE
    _CURRENT_REGISTER_SOURCE = source
    try:
        # --- (1) Top-level single-file Python plugins ---
        for py_file in sorted(hub_arch_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"_crucible_hub_arch_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                loaded.append(py_file.stem)
            except Exception as exc:
                log_warn(f"Failed to load architecture plugin {py_file.name}: {exc}")

        # --- (2) YAML spec plugins ---
        for yaml_file in sorted(hub_arch_dir.glob("*.yaml")):
            if yaml_file.name.startswith("_"):
                continue
            try:
                _register_from_spec_file(yaml_file.stem, yaml_file, source=source)
                loaded.append(yaml_file.stem)
            except Exception as exc:
                log_warn(f"Failed to load architecture spec {yaml_file.name}: {exc}")

        # --- (3) Directory-bundle Python plugins ---
        # For each subdirectory <name>/, try to load <name>/<name>.py if
        # present. Prepend the bundle's parent directory to sys.path briefly
        # so `import wm_base` (sibling module inside the same bundle) works
        # from within the primary file.
        if hub_arch_dir.is_dir():
            for sub in sorted(p for p in hub_arch_dir.iterdir() if p.is_dir()):
                if sub.name.startswith("_") or sub.name.startswith("."):
                    continue
                primary = sub / f"{sub.name}.py"
                if not primary.is_file():
                    continue
                module_name = f"_crucible_hub_arch_bundle_{sub.name}"
                bundle_parent_added = False
                try:
                    # Make siblings of primary importable by adding the
                    # BUNDLE dir (not the hub_arch_dir) to sys.path. This
                    # lets `import wm_base` inside code_wm.py resolve to
                    # the sibling wm_base.py.
                    bundle_parent = str(sub.resolve())
                    if bundle_parent not in sys.path:
                        sys.path.insert(0, bundle_parent)
                        bundle_parent_added = True
                    spec = importlib.util.spec_from_file_location(module_name, primary)
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    loaded.append(sub.name)
                except Exception as exc:
                    log_warn(
                        f"Failed to load architecture bundle {sub.name}: {exc}"
                    )
                finally:
                    if bundle_parent_added:
                        try:
                            sys.path.remove(bundle_parent)
                        except ValueError:
                            pass
    finally:
        _CURRENT_REGISTER_SOURCE = prev_source
    return loaded


def _register_from_spec_file(
    name: str,
    spec_path: Path,
    *,
    source: str = "local",
) -> None:
    """Load a YAML architecture spec and register it as a model family.

    This is the registry-level entry point used by ``load_global_architectures``
    and the architectures ``__init__`` loader.  For the full public API see
    ``crucible.models.composer.register_from_spec``.
    """
    from crucible.models.composer import register_from_spec
    register_from_spec(name, spec_path, source=source)


def reset_registry() -> None:
    """Clear all registered models (for testing only)."""
    _REGISTRY.clear()
    _REGISTRY_META.clear()
    _FAMILY_SCHEMAS.clear()
