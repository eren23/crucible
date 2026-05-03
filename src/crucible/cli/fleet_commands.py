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
        # FleetManager.status() returns a pre-rendered human-readable table
        # (via render_nodes). Print it directly — earlier versions of this
        # handler incorrectly called .get() on it as if it were a dict.
        print(fleet.status())

    elif cmd == "provision":
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        provision_kwargs: dict[str, object] = {
            "count": args.count,
            "name_prefix": getattr(args, "name_prefix", "crucible"),
        }
        # When --project is passed, load the project spec and apply its pod
        # overrides (image, gpu_type, container_disk, volume_disk,
        # interruptible, gpu_count). Mirrors what the MCP provision_project
        # tool already does so CLI users get the same pod shape.
        project = getattr(args, "project", "") or ""
        if project:
            from crucible.core.config import load_project_spec
            spec = load_project_spec(project, config.project_root)
            if spec.pod.image:
                provision_kwargs["image_name"] = spec.pod.image
            if spec.pod.gpu_type:
                provision_kwargs["gpu_type_ids"] = spec.pod.gpu_type
            if spec.pod.container_disk:
                provision_kwargs["container_disk_gb"] = spec.pod.container_disk
            if spec.pod.volume_disk:
                provision_kwargs["volume_gb"] = spec.pod.volume_disk
            if spec.pod.gpu_count:
                provision_kwargs["gpu_count"] = spec.pod.gpu_count
            # Spec interruptible only kicks in if --interruptible wasn't passed
            # explicitly. argparse defaults `interruptible` to False when the
            # flag is absent, so we use spec value as a fallback only when
            # the user did NOT pass --interruptible.
            if not getattr(args, "interruptible", False) and spec.pod.interruptible is not None:
                provision_kwargs["interruptible"] = spec.pod.interruptible
            elif getattr(args, "interruptible", False):
                provision_kwargs["interruptible"] = True
        elif getattr(args, "interruptible", False):
            provision_kwargs["interruptible"] = True

        nodes = fleet.provision(**provision_kwargs)
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
