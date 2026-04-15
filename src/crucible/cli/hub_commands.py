"""CLI handlers for the hub (crucible hub ...).

All hub operations use HubStore as the canonical storage backend.
"""
from __future__ import annotations

import argparse
import sys

from crucible.core.errors import StoreError
from crucible.core.hub import HubStore


def _resolve_hub_dir(args: argparse.Namespace) -> str | None:
    """Get explicit hub_dir from args, if provided."""
    return getattr(args, "hub_dir", None)


def _get_hub(args: argparse.Namespace) -> HubStore:
    """Load HubStore, handling the 'not initialized' case."""
    from crucible.core.config import load_config

    config = load_config()
    config_hub_dir = getattr(config, "hub_dir", None)
    hub_dir = HubStore.resolve_hub_dir(
        explicit=_resolve_hub_dir(args),
        config_hub_dir=config_hub_dir,
    )
    try:
        return HubStore(hub_dir)
    except StoreError:
        print("Hub not initialized. Run 'crucible hub init' first.", file=sys.stderr)
        sys.exit(1)


def handle_hub(args: argparse.Namespace) -> None:
    """Dispatch hub subcommands."""
    cmd = getattr(args, "hub_command", None)
    if cmd is None:
        print("Usage: crucible hub {init|link|sync|status}", file=sys.stderr)
        sys.exit(1)

    if cmd == "init":
        _cmd_init(args)
    elif cmd == "link":
        _cmd_link(args)
    elif cmd == "sync":
        _cmd_sync(args)
    elif cmd == "status":
        _cmd_status(args)
    else:
        print(f"Unknown hub command: {cmd}", file=sys.stderr)
        sys.exit(1)


def _cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new hub."""
    from crucible.core.config import load_config

    config = load_config()
    config_hub_dir = getattr(config, "hub_dir", None)
    hub_dir = HubStore.resolve_hub_dir(
        explicit=_resolve_hub_dir(args),
        config_hub_dir=config_hub_dir,
    )
    hub = HubStore.init(hub_dir)
    print(f"Hub initialized at {hub.hub_dir}")


def _cmd_link(args: argparse.Namespace) -> None:
    """Link the current project to the hub."""
    from crucible.core.config import load_config

    config = load_config()
    hub = _get_hub(args)
    hub.link_project(config.name, config.project_root)
    print(f"Linked project '{config.name}' at {config.project_root}")


def _cmd_sync(args: argparse.Namespace) -> None:
    """Sync hub state."""
    hub = _get_hub(args)
    summary = hub.sync()
    print(f"Hub sync complete:")
    print(f"  Projects: {summary['projects']}")
    print(f"  Tracks:   {summary['tracks']}")
    active = summary.get("active_track")
    if active:
        print(f"  Active:   {active}")
    print(f"  Status:   {summary['status']}")


def _cmd_status(args: argparse.Namespace) -> None:
    """Show hub status."""
    hub = _get_hub(args)

    projects = hub.list_projects()
    tracks = hub.list_tracks()
    active = hub.get_active_track()

    print(f"Hub: {hub.hub_dir}")
    print()

    print(f"Projects ({len(projects)}):")
    if projects:
        for p in projects:
            print(f"  {p['name']}  {p.get('project_root', '')}")
    else:
        print("  (none)")

    print()
    print(f"Tracks ({len(tracks)}):")
    if tracks:
        for t in tracks:
            marker = " *" if t.get("name") == active else ""
            desc = t.get("description", "")
            desc_str = f" - {desc}" if desc else ""
            print(f"  {t['name']}{marker}{desc_str}")
    else:
        print("  (none)")

    if active:
        print(f"\nActive track: {active}")
