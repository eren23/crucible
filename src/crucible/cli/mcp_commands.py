"""CLI handlers for MCP server commands."""
from __future__ import annotations

import argparse
import sys


def handle_mcp(args: argparse.Namespace) -> None:
    cmd = getattr(args, "mcp_command", None)

    if cmd == "serve":
        from crucible.mcp.server import main_cli

        main_cli()
    else:
        print("Usage: crucible mcp {serve}", file=sys.stderr)
