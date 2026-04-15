"""Centralised plugin discovery across the 3-tier hierarchy.

Scans ``~/.crucible-hub/plugins/{type}/`` (global) and
``.crucible/plugins/{type}/`` (local) for every registered
:class:`PluginRegistry`, importing any ``*.py`` files found.

Usage::

    from crucible.core.plugin_discovery import discover_all_plugins

    discover_all_plugins(
        registries={"optimizers": OPTIMIZER_REGISTRY, ...},
        project_root=Path("."),
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.plugin_registry import PluginRegistry

_DEFAULT_HUB_DIR = Path.home() / ".crucible-hub"


def discover_all_plugins(
    registries: dict[str, PluginRegistry[Any]],
    *,
    project_root: Path | None = None,
    hub_dir: Path | None = None,
    store_dir: str = ".crucible",
    plugins_subdir: str = "plugins",
) -> dict[str, list[str]]:
    """Scan global and local plugin directories for every registry.

    Parameters
    ----------
    registries:
        Mapping of directory-name -> PluginRegistry (e.g.
        ``{"optimizers": OPTIMIZER_REGISTRY}``).
    project_root:
        Project root for local plugin discovery.  Skipped when ``None``.
    hub_dir:
        Hub root (defaults to ``~/.crucible-hub``).
    store_dir:
        Project store directory name (default ``.crucible``).
    plugins_subdir:
        Subdirectory under store / hub for plugins (default ``plugins``).

    Returns
    -------
    dict mapping registry names to lists of successfully loaded plugin stems.
    """
    hub_dir = hub_dir or _DEFAULT_HUB_DIR
    loaded: dict[str, list[str]] = {}

    # Trigger builtin data source registrations (side-effect import).
    # Lives here — not at module level — because core/ must not depend on
    # non-core crucible modules.
    import crucible.data_sources  # noqa: F401

    # Merge data_sources registry so it is handled uniformly with all other plugin types
    from crucible.core.data_sources import _DATA_SOURCE_REGISTRY
    registries = {**registries, "data_sources": _DATA_SOURCE_REGISTRY}

    for dir_name, registry in registries.items():
        names: list[str] = []

        # Global tier — ~/.crucible-hub/plugins/{dir_name}/
        global_dir = hub_dir / plugins_subdir / dir_name
        names.extend(registry.load_plugins(global_dir, source="global"))

        # Local tier — .crucible/plugins/{dir_name}/
        if project_root is not None:
            local_dir = project_root / store_dir / plugins_subdir / dir_name
            names.extend(registry.load_plugins(local_dir, source="local"))

        loaded[dir_name] = names

    return loaded
