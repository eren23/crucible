"""CLI handlers for MCP server commands."""
from __future__ import annotations

import argparse
import os
import sys


def handle_mcp(args: argparse.Namespace) -> None:
    cmd = getattr(args, "mcp_command", None)

    if cmd == "serve":
        # Pass trace flags via environment variables so the server picks them up
        if getattr(args, "trace", False):
            os.environ["CRUCIBLE_TRACE"] = "1"
        trace_id = getattr(args, "trace_id", None)
        if trace_id:
            os.environ["CRUCIBLE_TRACE_ID"] = trace_id

        from crucible.mcp.server import main_cli

        main_cli()
    else:
        print("Usage: crucible mcp {serve}", file=sys.stderr)
