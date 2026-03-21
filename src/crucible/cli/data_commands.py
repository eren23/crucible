"""CLI handlers for data management commands."""
from __future__ import annotations

import argparse
import sys

from crucible.core.config import load_config


def handle_data(args: argparse.Namespace) -> None:
    config = load_config()
    cmd = getattr(args, "data_command", None)

    if cmd == "download":
        from crucible.data.download import DataManager

        dm = DataManager(config.data, project_root=config.project_root)
        variant = getattr(args, "variant", "sp1024")
        train_shards = getattr(args, "train_shards", 80)
        dm.download(variant=variant, train_shards=train_shards)
        print("Download complete.")

    elif cmd == "sync":
        from crucible.data.sync import sync_data_to_nodes
        from crucible.fleet.inventory import load_nodes

        nodes = load_nodes(config.project_root / config.nodes_file)
        sync_data_to_nodes(nodes, config)
        print("Data sync complete.")

    elif cmd == "status":
        from crucible.data.download import DataManager

        dm = DataManager(config.data, project_root=config.project_root)
        status = dm.status()
        print(f"Data root: {status.get('local_root', 'N/A')}")
        print(f"Variants available: {', '.join(status.get('variants', [])) or 'none'}")
        for variant, info in status.get("details", {}).items():
            print(f"  {variant}: {info.get('train_shards', 0)} train shards, {info.get('val_shards', 0)} val shards")

    else:
        print("Usage: crucible data {download|sync|status}", file=sys.stderr)
