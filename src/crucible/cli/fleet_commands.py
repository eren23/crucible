"""CLI handlers for fleet management commands."""
from __future__ import annotations

import argparse
import sys
import time

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError


def handle_fleet(args: argparse.Namespace) -> None:
    try:
        _handle_fleet(args)
    except CrucibleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _handle_fleet(args: argparse.Namespace) -> None:
    config = load_config()
    cmd = getattr(args, "fleet_command", None)

    if cmd == "status":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        status = fleet.status()
        print(f"Nodes: {status.get('nodes_total', 0)} total, {status.get('nodes_ready', 0)} ready")
        for node in status.get("nodes", []):
            print(f"  {node.get('name', '?')}: state={node.get('state', '?')} gpu={node.get('gpu', '?')}")

    elif cmd == "provision":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        nodes = fleet.provision(
            count=args.count,
            name_prefix=getattr(args, "name_prefix", "crucible"),
        )
        print(f"Provisioned {len(nodes)} nodes.")
        for n in nodes:
            print(f"  {n.get('name', '?')}: {n.get('gpu', '?')}")

    elif cmd == "destroy":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        node_list = getattr(args, "nodes", None)
        fleet.destroy(selected_names=set(node_list) if node_list else None)
        print("Nodes destroyed.")

    elif cmd == "bootstrap":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        fleet.bootstrap(
            train_shards=getattr(args, "train_shards", 1),
            skip_install=getattr(args, "skip_install", False),
            skip_data=getattr(args, "skip_data", False),
        )
        print("Bootstrap complete.")

    elif cmd == "sync":
        from crucible.fleet.inventory import load_nodes
        from crucible.fleet.sync import sync_repo

        nodes = load_nodes(config.project_root / config.nodes_file)
        for node in nodes:
            try:
                sync_repo(node, config.project_root, config.sync_excludes)
                print(f"  Synced {node['name']}")
            except CrucibleError as exc:
                print(f"  Failed {node['name']}: {exc}", file=sys.stderr)

    elif cmd == "monitor":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        watch_interval = getattr(args, "watch", 0)
        while True:
            output = fleet.monitor()
            if watch_interval > 0:
                print("\033[2J\033[H", end="")  # clear screen
            print(output)
            if watch_interval <= 0:
                break
            time.sleep(watch_interval)

    else:
        print("Usage: crucible fleet {status|provision|destroy|bootstrap|sync|monitor}", file=sys.stderr)
