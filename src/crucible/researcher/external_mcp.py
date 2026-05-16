"""External MCP consumption (Phase 3.5).

Lets Crucible call out to user-supplied MCP servers. Use cases the
plan called out:
- Spider Chat MCP for taste-curation
- Codex MCP for code mutations
- Any community MCP exposing domain tools (chemistry, bio, etc.)

Servers are declared in ``crucible.yaml`` under
``external_mcp.servers``::

    external_mcp:
      servers:
        spider:
          command: "spider-mcp"
          args: ["--mode", "stdio"]
          env: {SPIDER_TOKEN: "${SPIDER_TOKEN}"}
        codex:
          command: "codex"
          args: ["mcp", "serve"]

The two MCP tools shipped on top of this module:
- ``external_mcp_list_servers``: read-only enumeration + per-server
  tool listing.
- ``external_mcp_call``: invoke one tool on one server with JSON args.

Each call spawns a fresh subprocess (no persistent connection
pooling in MVP). Latency cost ~100-500ms per call; acceptable for
the autonomous-loop use case where these are infrequent reflection
hooks, not hot-path inference.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import sys
from contextlib import asynccontextmanager
from typing import Any

from crucible.core.errors import CrucibleError


class ExternalMCPError(CrucibleError):
    """External MCP server failed (subprocess, protocol, tool exec)."""


class ExternalMCPConfigError(CrucibleError):
    """The external_mcp config block is malformed."""


@asynccontextmanager
async def _connect(server_spec: dict[str, Any]):
    """Open a stdio MCP client to the configured server.

    Returns a connected ``ClientSession``. Caller is responsible for
    invoking tool calls within the ``async with`` block; the session
    + subprocess close on exit.
    """
    # Lazy-imported so the rest of Crucible doesn't need the mcp client
    # at import time. The mcp package is already a hard dep for the
    # server side, so this is essentially free.
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    command = server_spec.get("command")
    if not command:
        raise ExternalMCPConfigError("server spec missing required 'command' field.")
    args = server_spec.get("args", [])
    if isinstance(args, str):
        # Convenience: split a single string into argv via shlex.
        args = shlex.split(args)
    env = server_spec.get("env") or None
    if env:
        # Resolve ${VAR} placeholders against the current process env.
        env = {k: os.path.expandvars(str(v)) for k, v in env.items()}

    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# Default per-call timeout. Configurable via ``timeout`` field on the
# server spec or per-MCP-call ``timeout`` arg. Without a cap, a server
# that starts but never responds blocks the worker thread forever.
_DEFAULT_EXT_MCP_TIMEOUT = 30.0


def _spec_timeout(server_spec: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(server_spec.get("timeout", _DEFAULT_EXT_MCP_TIMEOUT))


async def _list_tools_async(
    server_spec: dict[str, Any], *, timeout: float = _DEFAULT_EXT_MCP_TIMEOUT,
) -> list[dict[str, Any]]:
    async with _connect(server_spec) as session:
        listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
        return [
            {
                "name": t.name,
                "description": (t.description or "").strip().splitlines()[0]
                if t.description else "",
            }
            for t in listed.tools
        ]


async def _call_tool_async(
    server_spec: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    timeout: float = _DEFAULT_EXT_MCP_TIMEOUT,
) -> dict[str, Any]:
    async with _connect(server_spec) as session:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, tool_args), timeout=timeout,
        )
        # MCP CallToolResult has a `content` list of TextContent /
        # ImageContent etc. Flatten to plain dicts for JSON transport.
        content = []
        for block in result.content:
            entry: dict[str, Any] = {"type": getattr(block, "type", "unknown")}
            text = getattr(block, "text", None)
            if text is not None:
                entry["text"] = text
            content.append(entry)
        return {
            "is_error": getattr(result, "isError", False),
            "content": content,
        }


def _run_async(coro):
    """Bridge: run an async coroutine from a sync MCP tool dispatcher.

    Safe because the parent MCP server routes sync handlers through
    ``asyncio.to_thread()`` — the handler executes in a thread-pool
    worker that has no running event loop of its own, so
    ``asyncio.run`` here creates a fresh loop without colliding.

    DO NOT call this from an async context directly — it will raise
    ``RuntimeError: asyncio.run() cannot be called from a running event
    loop``. Stay inside a thread or propagate ``async`` all the way up.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Public sync API used by the MCP dispatchers
# ---------------------------------------------------------------------------


def list_servers(config: Any) -> dict[str, Any]:
    """Read configured external MCP servers from project config.

    Returns ``{count, servers: [{name, command, args}]}``. Does not
    spawn subprocesses; safe to call cheaply for an enumeration probe.
    """
    raw = _read_servers_config(config)
    out = []
    for name, spec in raw.items():
        out.append({
            "name": name,
            "command": spec.get("command", ""),
            "args": spec.get("args", []),
            "has_env": bool(spec.get("env")),
        })
    return {"count": len(out), "servers": out}


def list_remote_tools(
    config: Any, server_name: str, *, timeout: float | None = None,
) -> dict[str, Any]:
    """Spawn the named server, enumerate its tools, shut down.

    ``timeout`` caps how long we'll wait for the server to respond to
    list_tools. Defaults to the per-spec ``timeout`` or 30s.
    """
    spec = _resolve_server(config, server_name)
    t = _spec_timeout(spec, timeout)
    try:
        tools = _run_async(_list_tools_async(spec, timeout=t))
    except Exception as exc:
        raise ExternalMCPError(
            f"external_mcp[{server_name}]: list_tools failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return {"server": server_name, "count": len(tools), "tools": tools}


def call_remote_tool(
    config: Any,
    server_name: str,
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Invoke a tool on the named external server.

    ``timeout`` caps how long we'll wait for the server's tool call to
    return. Defaults to the per-spec ``timeout`` or 30s. A hanging
    server raises ExternalMCPError(asyncio.TimeoutError) — the worker
    thread does not block forever.
    """
    spec = _resolve_server(config, server_name)
    t = _spec_timeout(spec, timeout)
    try:
        result = _run_async(
            _call_tool_async(spec, tool_name, dict(tool_args or {}), timeout=t)
        )
    except Exception as exc:
        raise ExternalMCPError(
            f"external_mcp[{server_name}].{tool_name}: call failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return {
        "server": server_name,
        "tool": tool_name,
        **result,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_servers_config(config: Any) -> dict[str, dict[str, Any]]:
    """Pull ``external_mcp.servers`` from the project config.

    Tolerates absent config — returns ``{}`` so callers can probe
    without raising. The config layer hasn't been extended with a
    typed ``external_mcp`` block yet (Phase 4 candidate); for MVP we
    read from a raw attribute or a yaml fallback.
    """
    block = getattr(config, "external_mcp", None)
    if block is None:
        # Try a yaml side-load — the project.yaml may carry the block
        # even when ProjectConfig doesn't model it yet.
        try:
            import yaml
            yaml_path = config.project_root / "crucible.yaml"
            if yaml_path.exists():
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                block = data.get("external_mcp", {})
        except Exception:
            block = {}
    if not isinstance(block, dict):
        return {}
    servers = block.get("servers") if isinstance(block, dict) else None
    if not isinstance(servers, dict):
        return {}
    return servers


def _resolve_server(config: Any, name: str) -> dict[str, Any]:
    """Look up one server spec by name; raise if missing/malformed."""
    servers = _read_servers_config(config)
    spec = servers.get(name)
    if not isinstance(spec, dict):
        raise ExternalMCPConfigError(
            f"No external_mcp server registered as {name!r}. "
            f"Configured: {sorted(servers.keys())}"
        )
    return spec
