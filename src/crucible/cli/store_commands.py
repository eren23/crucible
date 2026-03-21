"""CLI handlers for the version store (crucible store ...)."""
from __future__ import annotations

import argparse
import json
import sys

from crucible.core.config import load_config


def handle_store(args: argparse.Namespace) -> None:
    """Dispatch store subcommands."""
    cmd = getattr(args, "store_command", None)
    if cmd is None:
        print("Usage: crucible store {list|show|history|commit}", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    store_dir = config.project_root / config.store_dir

    from crucible.core.store import VersionStore

    store = VersionStore(store_dir)

    if cmd == "list":
        _cmd_list(args, store)
    elif cmd == "show":
        _cmd_show(args, store)
    elif cmd == "history":
        _cmd_history(args, store)
    elif cmd == "commit":
        _cmd_commit(args, store)
    else:
        print(f"Unknown store command: {cmd}", file=sys.stderr)
        sys.exit(1)


def _cmd_list(args: argparse.Namespace, store) -> None:
    """List resources of a given type."""
    resource_type = args.resource_type
    type_map = {"designs": "experiment_design", "context": "research_context"}
    rtype = type_map.get(resource_type, resource_type)

    resources = store.list_resources(rtype)
    if not resources:
        print(f"No {resource_type} found.")
        return

    for meta in resources:
        name = meta["resource_name"]
        ver = meta["version"]
        summary = meta.get("summary", "")[:60]
        created = meta.get("created_at", "")[:19]
        print(f"  {name} (v{ver})  {created}  {summary}")


def _cmd_show(args: argparse.Namespace, store) -> None:
    """Show the current version of a resource."""
    path = args.resource_path
    if "/" not in path:
        print("Usage: crucible store show {type}/{name}", file=sys.stderr)
        sys.exit(1)

    rtype, rname = path.split("/", 1)
    type_map = {"designs": "experiment_design", "context": "research_context"}
    rtype = type_map.get(rtype, rtype)

    result = store.get_current(rtype, rname)
    if result is None:
        print(f"Not found: {path}")
        sys.exit(1)

    meta, content = result
    print(f"Version: {meta['version']}  Created: {meta.get('created_at', '')[:19]}")
    print(f"By: {meta.get('created_by', 'unknown')}  Summary: {meta.get('summary', '')}")
    print("---")
    import yaml

    print(yaml.dump(content, default_flow_style=False, sort_keys=False))


def _cmd_history(args: argparse.Namespace, store) -> None:
    """Show version history for a resource."""
    path = args.resource_path
    if "/" not in path:
        print("Usage: crucible store history {type}/{name}", file=sys.stderr)
        sys.exit(1)

    rtype, rname = path.split("/", 1)
    type_map = {"designs": "experiment_design", "context": "research_context"}
    rtype = type_map.get(rtype, rtype)

    versions = store.history(rtype, rname)
    if not versions:
        print(f"No history for: {path}")
        return

    for meta in versions:
        ver = meta["version"]
        created = meta.get("created_at", "")[:19]
        by = meta.get("created_by", "unknown")
        summary = meta.get("summary", "")[:60]
        git = " [committed]" if meta.get("git_committed") else ""
        print(f"  v{ver}  {created}  by:{by}  {summary}{git}")


def _cmd_commit(args: argparse.Namespace, store) -> None:
    """Git commit all uncommitted versions."""
    committed = 0
    for versions in store._index.values():
        for meta in versions:
            if not meta.get("git_committed"):
                sha = store.git_commit_version(meta)
                if sha:
                    print(f"  Committed {meta['version_id']} -> {sha[:8]}")
                    committed += 1

    if committed:
        print(f"Committed {committed} versions.")
    else:
        print("Nothing to commit.")
