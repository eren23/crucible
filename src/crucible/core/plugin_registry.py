"""Generic plugin registry with 3-tier precedence and file-based discovery.

Every pluggable component in Crucible (optimizers, schedulers, fleet providers,
composer extensions, loggers, callbacks) uses this as its foundation.  The
pattern mirrors the model architecture registry in ``models/registry.py`` but
is generalised so each subsystem gets an identical API surface.

Precedence tiers (lowest to highest):
    builtin  — shipped with Crucible core
    global   — installed in ``~/.crucible-hub/plugins/{type}/``
    local    — project-level ``.crucible/plugins/{type}/``

Higher-precedence registrations silently override lower ones; equal-or-lower
precedence raises ``PluginError``.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from crucible.core.errors import PluginError

T = TypeVar("T")

PRECEDENCE: dict[str, int] = {"builtin": 0, "global": 1, "local": 2}
_VALID_SOURCES = frozenset(PRECEDENCE)


class PluginRegistry(Generic[T]):
    """Thread-safe plugin registry with 3-tier precedence.

    Parameters
    ----------
    plugin_type:
        Human-readable name used in error messages and discovery paths
        (e.g. ``"optimizer"``, ``"scheduler"``).
    """

    def __init__(self, plugin_type: str) -> None:
        self.plugin_type = plugin_type
        self._registry: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        factory: Any,
        *,
        source: str = "builtin",
    ) -> None:
        """Register a plugin under *name*.

        *factory* is typically a class or callable that produces the plugin
        instance.  *source* must be one of ``"builtin"``, ``"global"``, or
        ``"local"``.

        Higher-precedence sources silently override lower ones.  Attempting
        to register with equal-or-lower precedence when a higher source is
        already present raises ``PluginError``.
        """
        if source not in _VALID_SOURCES:
            raise PluginError(
                f"Invalid source {source!r} for {self.plugin_type} plugin "
                f"{name!r}; must be one of {sorted(_VALID_SOURCES)}"
            )
        with self._lock:
            if name in self._registry:
                existing_source = self._meta.get(name, {}).get("source", "builtin")
                if PRECEDENCE[source] <= PRECEDENCE[existing_source]:
                    raise PluginError(
                        f"{self.plugin_type} plugin {name!r} is already registered "
                        f"(source: {existing_source})"
                    )
            self._registry[name] = factory
            self._meta[name] = {"source": source}

    # ------------------------------------------------------------------
    # Build / Lookup
    # ------------------------------------------------------------------

    def build(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered plugin by *name*.

        All *kwargs* are forwarded to the factory / class constructor.
        """
        with self._lock:
            factory = self._registry.get(name)
        if factory is None:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise PluginError(
                f"Unknown {self.plugin_type} plugin {name!r}. "
                f"Registered: {available}"
            )
        return factory(**kwargs)

    def get(self, name: str) -> Any | None:
        """Return the raw factory/class for *name*, or ``None``."""
        with self._lock:
            return self._registry.get(name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_plugins(self) -> list[str]:
        """Return sorted list of registered plugin names."""
        with self._lock:
            return sorted(self._registry)

    def list_plugins_detailed(self) -> list[dict[str, str]]:
        """Return plugins with source metadata."""
        with self._lock:
            return sorted(
                [
                    {
                        "name": name,
                        "source": self._meta.get(name, {}).get("source", "unknown"),
                    }
                    for name in self._registry
                ],
                key=lambda d: d["name"],
            )

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._registry

    def __len__(self) -> int:
        with self._lock:
            return len(self._registry)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def register_schema(self, name: str, schema: dict[str, Any]) -> None:
        """Attach a parameter schema to a named plugin."""
        with self._lock:
            self._schemas[name] = schema

    def get_schema(self, name: str) -> dict[str, Any]:
        """Return the parameter schema for *name*, or a stub."""
        with self._lock:
            return self._schemas.get(
                name, {"note": f"No schema registered for {self.plugin_type} {name!r}"}
            )

    # ------------------------------------------------------------------
    # File-based discovery
    # ------------------------------------------------------------------

    def load_plugins(self, directory: Path, *, source: str = "local") -> list[str]:
        """Import all ``*.py`` files from *directory* and return loaded names.

        Each Python file is expected to call ``register()`` on this registry
        (or the convenience wrapper) at module level.  Files starting with
        ``_`` are skipped.
        """
        if not directory.is_dir():
            return []

        loaded: list[str] = []
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"_crucible_plugin_{self.plugin_type}_{source}_{py_file.stem}"
            # Guard against double-execution on repeated discovery calls
            if module_name in sys.modules:
                loaded.append(py_file.stem)
                continue
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                loaded.append(py_file.stem)
            except PluginError:
                raise  # Propagate registration conflicts
            except Exception as exc:
                # Remove failed module from sys.modules to allow retry
                sys.modules.pop(module_name, None)
                from crucible.core.log import log_warn
                log_warn(f"Failed to load {self.plugin_type} plugin {py_file.name}: {exc}")
        return loaded

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def unregister(self, name: str) -> None:
        """Remove a single plugin by name (thread-safe, for testing)."""
        with self._lock:
            self._registry.pop(name, None)
            self._meta.pop(name, None)
            self._schemas.pop(name, None)

    def reset(self) -> None:
        """Clear all registrations (for testing only)."""
        with self._lock:
            self._registry.clear()
            self._meta.clear()
            self._schemas.clear()
