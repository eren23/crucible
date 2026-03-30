"""Reusable launcher bundle resolution for external projects.

Launcher bundles let external-project specs reuse a named, shareable entrypoint
instead of copying one-off scripts into the remote workspace. Bundles may live
in the project store, the installed hub plugins directory, or directly inside a
configured tap clone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def launcher_runtime_dir(workspace: str, name: str) -> str:
    """Return the remote runtime directory for a launcher bundle."""
    return f"{workspace.rstrip('/')}/.crucible/launchers/{name}"


def launcher_runtime_entry(workspace: str, name: str, entry: str) -> str:
    """Return the remote runtime entrypoint path for a launcher bundle."""
    return f"{launcher_runtime_dir(workspace, name)}/{entry}"


def resolve_launcher_bundle(
    *,
    project_root: Path,
    launcher_name: str,
    hub_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Resolve a launcher bundle from local, installed hub, or tap sources."""
    if not launcher_name:
        return None

    from crucible.core.hub import HubStore

    roots: list[tuple[str, Path, dict[str, Any]]] = []
    roots.append((
        "local",
        project_root / ".crucible" / "plugins" / "launchers" / launcher_name,
        {},
    ))

    resolved_hub_dir = hub_dir or HubStore.resolve_hub_dir()
    if resolved_hub_dir is not None:
        roots.append((
            "hub",
            resolved_hub_dir / "plugins" / "launchers" / launcher_name,
            {},
        ))
        taps_root = resolved_hub_dir / "taps"
        if taps_root.is_dir():
            for tap_dir in sorted(taps_root.iterdir()):
                roots.append((
                    "tap",
                    tap_dir / "launchers" / launcher_name,
                    {"tap": tap_dir.name},
                ))

    for source, bundle_dir, extra in roots:
        if not bundle_dir.is_dir():
            continue
        manifest_path = bundle_dir / "plugin.yaml"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                manifest = loaded
        entry = str(manifest.get("entry", f"{launcher_name}.py"))
        return {
            "name": launcher_name,
            "source": source,
            "path": bundle_dir,
            "entry": entry,
            "manifest": manifest,
            **extra,
        }
    return None
