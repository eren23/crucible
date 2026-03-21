"""CLI handlers for fleet management commands."""
from __future__ import annotations

import argparse
import sys

from crucible.core.config import load_config


def handle_fleet(args: argparse.Namespace) -> None:
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

    elif cmd == "destroy":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        fleet.destroy(node_names=getattr(args, "nodes", None))
        print("Nodes destroyed.")

    elif cmd == "bootstrap":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        fleet.bootstrap()
        print("Bootstrap complete.")

    elif cmd == "sync":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        from crucible.fleet.inventory import load_nodes
        from crucible.fleet.sync import sync_repo

        nodes = load_nodes(config.project_root / config.nodes_file)
        for node in nodes:
            try:
                sync_repo(node, config.project_root, config.sync_excludes)
                print(f"  Synced {node['name']}")
            except Exception as exc:
                print(f"  Failed {node['name']}: {exc}", file=sys.stderr)

    elif cmd == "monitor":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        status = fleet.status()
        print(f"Nodes: {status.get('nodes_total', 0)} total")
        for node in status.get("nodes", []):
            print(f"  {node.get('name', '?')}: {node.get('state', '?')}")

    else:
        print("Usage: crucible fleet {status|provision|destroy|bootstrap|sync|monitor}", file=sys.stderr)
