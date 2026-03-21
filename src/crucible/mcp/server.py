"""MCP server exposing Crucible fleet operations as tools for Claude agents.

Run via stdio:
    crucible mcp serve
    python -m crucible.mcp.server
"""
from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from crucible.mcp.tools import TOOL_DISPATCH

app = Server("crucible-fleet")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_text(obj: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(obj, indent=2, default=str))]


def _error_text(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> list[TextContent]:
    try:
        result = fn(*args, **kwargs)
        return _json_text(result)
    except Exception:
        return _error_text(traceback.format_exc())


# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="get_fleet_status",
        description="Node inventory, health summary, and current assignments.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_leaderboard",
        description="Top N experiment results sorted by primary metric (lower is better).",
        inputSchema={
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Number of results to return.", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_queue_status",
        description="Fleet queue state: counts of queued, running, and completed experiments.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="enqueue_experiment",
        description="Add an experiment configuration to the fleet queue.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short experiment name (no spaces)."},
                "config": {
                    "type": "object",
                    "description": "Environment variable overrides.",
                    "additionalProperties": {"type": "string"},
                },
                "tier": {"type": "string", "description": "Experiment tier.", "default": "proxy"},
                "backend": {"type": "string", "description": "Training backend.", "default": "torch"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags.", "default": []},
            },
            "required": ["name", "config"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_experiment_result",
        description="Get the result for a specific experiment run_id.",
        inputSchema={
            "type": "object",
            "properties": {"run_id": {"type": "string", "description": "The unique run identifier."}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="provision_nodes",
        description="Create N new compute nodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of nodes to create.", "default": 2},
                "name_prefix": {"type": "string", "description": "Node name prefix.", "default": "crucible"},
            },
            "required": ["count"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="destroy_nodes",
        description="Tear down tracked nodes. Optionally specify node names.",
        inputSchema={
            "type": "object",
            "properties": {
                "node_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node names to destroy. If empty, destroys all.",
                    "default": [],
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="sync_code",
        description="Push local code to nodes via rsync.",
        inputSchema={
            "type": "object",
            "properties": {
                "node_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node names to sync. If empty, syncs to all.",
                    "default": [],
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_research_state",
        description="Current research state: hypotheses, beliefs, budget.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_sensitivity",
        description="Parameter sensitivity analysis across completed experiments.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[type-arg]
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return _error_text(f"Unknown tool: {name}")
    return _safe_call(handler, arguments)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def _run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main_cli() -> None:
    """Entry point for crucible-mcp console script."""
    asyncio.run(_run_server())


if __name__ == "__main__":
    main_cli()
