"""CLI handlers for MCP server commands."""
from __future__ import annotations

import argparse
import json
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
        return

    if cmd == "call":
        _handle_mcp_call(args)
        return

    print("Usage: crucible mcp {serve|call}", file=sys.stderr)
    sys.exit(2)


def _handle_mcp_call(args: argparse.Namespace) -> None:
    """Invoke one MCP tool and print its JSON return value.

    Bypasses the stdio server — calls the in-process TOOL_DISPATCH table
    directly. Useful for the 5-minute quickstart and any operator who
    wants a one-shot tool call without spinning up the full MCP loop.
    """
    tool_name = args.tool_name
    try:
        tool_args = json.loads(args.args)
    except json.JSONDecodeError as exc:
        print(f"--args must be a JSON object: {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(tool_args, dict):
        print(f"--args must decode to a JSON object, got {type(tool_args).__name__}",
              file=sys.stderr)
        sys.exit(2)

    from crucible.mcp.tools import TOOL_DISPATCH

    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        print(f"Unknown MCP tool: {tool_name!r}", file=sys.stderr)
        print(
            f"Run `crucible mcp call <name>` with one of "
            f"{len(TOOL_DISPATCH)} registered tools "
            f"(see docs/mcp-tools.md or src/crucible/mcp/server.py for the list).",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        result = handler(tool_args)
    except Exception as exc:
        print(f"Tool {tool_name!r} raised: [{type(exc).__name__}] {exc}",
              file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
