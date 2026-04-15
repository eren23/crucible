"""CLI handlers for session trace viewing and export (crucible trace ...)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crucible.core.config import load_config


def _get_traces_dir() -> Path:
    config = load_config()
    return config.project_root / ".crucible" / "traces"


def handle_trace(args: argparse.Namespace) -> None:
    """Dispatch trace subcommands."""
    cmd = getattr(args, "trace_command", None)
    if cmd is None:
        print("Usage: crucible trace {list|show|export}", file=sys.stderr)
        sys.exit(1)

    if cmd == "list":
        _cmd_list(args)
    elif cmd == "show":
        _cmd_show(args)
    elif cmd == "export":
        _cmd_export(args)
    else:
        print(f"Unknown trace command: {cmd}", file=sys.stderr)
        sys.exit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    """List all traces in .crucible/traces/."""
    from crucible.mcp.tracer import load_trace_meta

    traces_dir = _get_traces_dir()
    if not traces_dir.exists():
        print("No traces directory found. Run with --trace to generate traces.")
        return

    meta_files = sorted(traces_dir.glob("*.meta.yaml"))
    if not meta_files:
        print("No traces found.")
        return

    print(f"{'SESSION ID':<28} {'STARTED':<22} {'TOOL CALLS':>10}")
    print("-" * 62)
    for meta_path in meta_files:
        try:
            meta = load_trace_meta(meta_path)
        except (OSError, ValueError):
            continue
        session_id = meta.get("session_id", meta_path.stem.replace(".meta", ""))
        started = meta.get("started_at", "")[:19]
        tool_calls = meta.get("tool_calls", 0)
        print(f"{session_id:<28} {started:<22} {tool_calls:>10}")


def _cmd_show(args: argparse.Namespace) -> None:
    """Print the JSONL entries for a session, pretty-printed."""
    from crucible.mcp.tracer import load_trace

    traces_dir = _get_traces_dir()
    session_id = args.session_id
    trace_path = traces_dir / f"{session_id}.jsonl"

    if not trace_path.exists():
        print(f"Trace not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_trace(trace_path)
    if not entries:
        print("Trace is empty.")
        return

    for entry in entries:
        print(json.dumps(entry, indent=2, default=str))
        print()


def _cmd_export(args: argparse.Namespace) -> None:
    """Export a trace as a shareable markdown document."""
    from crucible.mcp.tracer import load_trace, load_trace_meta

    traces_dir = _get_traces_dir()
    session_id = args.session_id
    trace_path = traces_dir / f"{session_id}.jsonl"
    meta_path = traces_dir / f"{session_id}.meta.yaml"

    if not trace_path.exists():
        print(f"Trace not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_trace(trace_path)

    # Load meta if available, otherwise derive from entries
    if meta_path.exists():
        meta = load_trace_meta(meta_path)
    else:
        meta = {
            "session_id": session_id,
            "started_at": entries[0]["ts"] if entries else "unknown",
            "ended_at": entries[-1]["ts"] if entries else "unknown",
            "tool_calls": len(entries),
        }

    started = meta.get("started_at", "unknown")
    ended = meta.get("ended_at", "unknown")
    tool_calls = meta.get("tool_calls", len(entries))

    lines: list[str] = []
    lines.append(f"# Crucible Session Trace: {session_id}")
    lines.append("")
    lines.append(f"**Started:** {started}  ")
    lines.append(f"**Ended:** {ended}  ")
    lines.append(f"**Tool calls:** {tool_calls}")
    lines.append("")
    lines.append("## Tool Call Sequence")

    for i, entry in enumerate(entries, 1):
        tool = entry.get("tool", "unknown")
        duration_ms = entry.get("duration_ms", 0)
        duration_s = duration_ms / 1000.0
        status = entry.get("status", "ok")

        status_suffix = ""
        if status != "ok":
            status_suffix = f" [{status}]"

        lines.append("")
        lines.append(f"### {i}. `{tool}` ({duration_s:.1f}s){status_suffix}")

        # Arguments
        arguments = entry.get("arguments", {})
        lines.append("**Arguments:**")
        lines.append("```json")
        lines.append(json.dumps(arguments, indent=2, default=str))
        lines.append("```")

        # Error (if any)
        error = entry.get("error")
        if error:
            lines.append("**Error:**")
            lines.append(f"```\n{error}\n```")

        # Result
        result = entry.get("result")
        if result is not None:
            lines.append("**Result:**")
            lines.append("```json")
            if isinstance(result, (dict, list)):
                lines.append(json.dumps(result, indent=2, default=str))
            else:
                lines.append(str(result))
            lines.append("```")

    output = "\n".join(lines) + "\n"

    # Write to file or stdout
    out_path = getattr(args, "output", None)
    if out_path:
        Path(out_path).write_text(output, encoding="utf-8")
        print(f"Exported to {out_path}")
    else:
        print(output)
