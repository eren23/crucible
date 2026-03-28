"""CLI handlers for community taps (crucible tap ...).

Manages plugin taps: adding, removing, searching, installing, and
publishing plugins from git-based community repositories.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from crucible.core.errors import CrucibleError, TapError


def _get_tap_manager(args: argparse.Namespace) -> Any:
    """Resolve hub dir and return a TapManager."""
    from crucible.core.config import load_config
    from crucible.core.hub import HubStore
    from crucible.core.tap import TapManager

    config = load_config()
    hub_dir = HubStore.resolve_hub_dir(config_hub_dir=getattr(config, "hub_dir", None))
    try:
        HubStore(hub_dir)  # validate hub is initialized
    except CrucibleError:
        print("Hub not initialized. Run 'crucible hub init' first.", file=sys.stderr)
        sys.exit(1)
    return TapManager(hub_dir)


def handle_tap(args: argparse.Namespace) -> None:
    """Dispatch tap subcommands."""
    cmd = getattr(args, "tap_command", None)
    if cmd is None:
        print("Usage: crucible tap {add|remove|list|sync|search|install|uninstall|installed|publish|info}", file=sys.stderr)
        sys.exit(1)

    try:
        if cmd == "add":
            _cmd_add(args)
        elif cmd == "remove":
            _cmd_remove(args)
        elif cmd == "list":
            _cmd_list(args)
        elif cmd == "sync":
            _cmd_sync(args)
        elif cmd == "search":
            _cmd_search(args)
        elif cmd == "install":
            _cmd_install(args)
        elif cmd == "uninstall":
            _cmd_uninstall(args)
        elif cmd == "installed":
            _cmd_installed(args)
        elif cmd == "publish":
            _cmd_publish(args)
        elif cmd == "info":
            _cmd_info(args)
        elif cmd == "push":
            _cmd_push(args)
        elif cmd == "submit-pr":
            _cmd_submit_pr(args)
        else:
            print(f"Unknown tap command: {cmd}", file=sys.stderr)
            sys.exit(1)
    except CrucibleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_add(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    result = tm.add_tap(args.url, name=getattr(args, "name", "") or "")
    print(f"Added tap: {result['name']} ({result['url']})")


def _cmd_remove(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    tm.remove_tap(args.name)
    print(f"Removed tap: {args.name}")


def _cmd_list(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    taps = tm.list_taps()
    if not taps:
        print("No taps configured. Add one with: crucible tap add <url>")
        return
    for tap in taps:
        synced = tap.get("last_synced", "never")
        print(f"  {tap['name']}  {tap['url']}  (synced: {synced})")


def _cmd_sync(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    result = tm.sync_tap(getattr(args, "name", "") or "")
    for name in result.get("synced", []):
        print(f"  synced: {name}")
    for err in result.get("errors", []):
        print(f"  error: {err}", file=sys.stderr)


def _cmd_search(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    results = tm.search(args.query, plugin_type=getattr(args, "type", "") or "")
    if not results:
        print(f"No plugins found for '{args.query}'")
        return
    for r in results:
        tags = ", ".join(r.get("tags", []))
        print(f"  {r['name']}  ({r['type']})  v{r.get('version', '?')}  [{r['tap']}]")
        if r.get("description"):
            print(f"    {r['description']}")
        if tags:
            print(f"    tags: {tags}")


def _cmd_install(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    result = tm.install(args.name, tap=getattr(args, "tap", "") or "")
    print(f"Installed: {result['name']} ({result['type']}) v{result.get('version', '?')} from [{result['tap']}]")
    print(f"  -> {result['path']}")


def _cmd_uninstall(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    tm.uninstall(args.name)
    print(f"Uninstalled: {args.name}")


def _cmd_installed(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    packages = tm.list_installed(plugin_type=getattr(args, "type", "") or "")
    if not packages:
        print("No tap plugins installed.")
        return
    for p in packages:
        print(f"  {p['name']}  ({p['type']})  v{p.get('version', '?')}  [{p['tap']}]  installed: {p.get('installed_at', '?')}")


def _cmd_publish(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    result = tm.publish(args.name, args.type, args.tap)
    print(f"Published: {result['name']} ({result['type']}) to [{result['tap']}]")
    for step in result.get("next_steps", []):
        print(f"  {step}")


def _cmd_info(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    info = tm.get_package_info(args.name)
    if info is None:
        print(f"Package '{args.name}' not found in any tap.")
        return
    print(f"  Name: {info.get('name')}")
    print(f"  Type: {info.get('type')}")
    print(f"  Version: {info.get('version', '?')}")
    print(f"  Author: {info.get('author', '?')}")
    print(f"  Description: {info.get('description', '')}")
    print(f"  Tap: {info.get('_tap', '?')}")
    print(f"  Installed: {'yes' if info.get('installed') else 'no'}")
    if info.get("benchmarks"):
        print(f"  Benchmarks: {info['benchmarks']}")
    if info.get("tags"):
        print(f"  Tags: {', '.join(info['tags'])}")


def _cmd_push(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    result = tm.push(args.tap)
    print(f"Pushed tap: {result['tap']}")


def _cmd_submit_pr(args: argparse.Namespace) -> None:
    tm = _get_tap_manager(args)
    result = tm.submit_pr(
        args.tap,
        title=getattr(args, "title", "") or "",
        body=getattr(args, "body", "") or "",
    )
    if result["status"] == "pr_created":
        print(f"PR created: {result['pr_url']}")
    elif result["status"] == "manual":
        print(result["instructions"])
    else:
        print(f"Error: {result.get('error', 'unknown')}")
        print(result.get("instructions", ""))
