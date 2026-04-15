"""CLI handlers for experiment tracks (crucible track ...).

All track operations use HubStore as the canonical storage backend.
"""
from __future__ import annotations

import argparse
import sys

import yaml

from crucible.core.errors import StoreError
from crucible.core.hub import HubStore


def _get_hub(args: argparse.Namespace) -> HubStore:
    """Load HubStore, handling the 'not initialized' case."""
    from crucible.core.config import load_config

    config = load_config()
    config_hub_dir = getattr(config, "hub_dir", None)
    hub_dir = HubStore.resolve_hub_dir(
        explicit=getattr(args, "hub_dir", None),
        config_hub_dir=config_hub_dir,
    )
    try:
        return HubStore(hub_dir)
    except StoreError:
        print("Hub not initialized. Run 'crucible hub init' first.", file=sys.stderr)
        sys.exit(1)


def handle_track(args: argparse.Namespace) -> None:
    """Dispatch track subcommands."""
    cmd = getattr(args, "track_command", None)
    if cmd is None:
        print("Usage: crucible track {create|list|switch|show}", file=sys.stderr)
        sys.exit(1)

    if cmd == "create":
        _cmd_create(args)
    elif cmd == "list":
        _cmd_list(args)
    elif cmd == "switch":
        _cmd_switch(args)
    elif cmd == "show":
        _cmd_show(args)
    else:
        print(f"Unknown track command: {cmd}", file=sys.stderr)
        sys.exit(1)


def _cmd_create(args: argparse.Namespace) -> None:
    """Create a new experiment track."""
    hub = _get_hub(args)
    name = args.name
    description = getattr(args, "description", "") or ""
    tags = getattr(args, "tags", None) or []

    try:
        hub.create_track(name, description=description, tags=tags)
        print(f"Track '{name}' created.")
        if description:
            print(f"  Description: {description}")
    except StoreError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    """List all experiment tracks."""
    hub = _get_hub(args)
    tracks = hub.list_tracks()
    active = hub.get_active_track()

    if not tracks:
        print("No tracks found.")
        return

    for t in tracks:
        name = t.get("name", "")
        marker = " *" if name == active else ""
        desc = t.get("description", "")
        status = t.get("status", "")
        tags = ", ".join(t.get("tags", []))
        desc_str = f"  {desc}" if desc else ""
        tag_str = f"  [{tags}]" if tags else ""
        status_str = f"  ({status})" if status else ""
        print(f"  {name}{marker}{status_str}{desc_str}{tag_str}")


def _cmd_switch(args: argparse.Namespace) -> None:
    """Switch the active experiment track."""
    hub = _get_hub(args)
    name = args.name

    try:
        hub.activate_track(name)
        print(f"Switched to track '{name}'.")
    except StoreError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_show(args: argparse.Namespace) -> None:
    """Show details of a specific track."""
    hub = _get_hub(args)
    name = args.name

    track = hub.get_track(name)
    if track is None:
        print(f"Track '{name}' not found.", file=sys.stderr)
        sys.exit(1)

    active = hub.get_active_track()
    if name == active:
        print(f"Track: {name} (active)")
    else:
        print(f"Track: {name}")

    print(yaml.dump(track, default_flow_style=False, sort_keys=False))
