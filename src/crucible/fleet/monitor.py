"""Node status probing, render functions, live monitoring."""
from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crucible.core.types import NodeRecord
from crucible.fleet.inventory import ready_state
from crucible.fleet.sync import remote_exec


# ---------------------------------------------------------------------------
# Status probing
# ---------------------------------------------------------------------------

ACTIVE_STATES = frozenset(
    {"queued", "starting", "warming_up", "training", "validating",
     "serializing", "running", "finalizing"}
)


def probe_node_status(node: NodeRecord) -> dict[str, Any]:
    """SSH into a node and read the most recent ``*.status.json``."""
    workspace = node.get("workspace_path", "/workspace/project")
    py = node.get("python_bin", "python3")
    command = (
        f"cd {shlex.quote(workspace)} && {shlex.quote(py)} - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        "paths = sorted(Path('logs').glob('*.status.json'), "
        "key=lambda p: p.stat().st_mtime, reverse=True)\n"
        "if not paths:\n"
        "    print(json.dumps({'state': 'no_status'}))\n"
        "else:\n"
        "    print(paths[0].read_text())\n"
        "PY"
    )
    proc = remote_exec(node, command, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"state": "unreachable", "error": (proc.stderr or "").strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"state": "invalid_status", "error": proc.stdout[:200]}


# ---------------------------------------------------------------------------
# Rendering helpers (local status files)
# ---------------------------------------------------------------------------

def parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO timestamp, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def age_seconds(status: dict[str, Any]) -> float | None:
    """Seconds since the last heartbeat / update in a status dict."""
    ts = parse_ts(status.get("last_heartbeat_at") or status.get("updated_at"))
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds()


def load_statuses(logs_dir: Path) -> list[dict[str, Any]]:
    """Load all ``*.status.json`` files from a directory."""
    statuses: list[dict[str, Any]] = []
    for path in sorted(logs_dir.glob("*.status.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            statuses.append({
                "run_id": path.stem,
                "state": "corrupt",
                "status_path": str(path),
                "error": "invalid_json",
            })
    statuses.sort(
        key=lambda s: s.get("updated_at") or s.get("created_at") or "",
        reverse=True,
    )
    return statuses


def render_statuses(
    statuses: list[dict[str, Any]],
    stale_seconds: int,
    show_all: bool,
) -> str:
    """Render a table of experiment statuses."""
    lines = [
        f"{'Run ID':<18} {'State':<18} {'Phase':<12} {'Age(s)':<8} "
        f"{'Step':<8} {'Metric':<18} {'Backend':<8}"
    ]
    lines.append("-" * 100)
    for status in statuses:
        age = age_seconds(status)
        if (
            not show_all
            and status.get("state") not in ACTIVE_STATES
            and (age is None or age > stale_seconds)
        ):
            continue
        metric = "-"
        if status.get("latest_val_loss") is not None:
            metric = f"val_loss={status['latest_val_loss']:.4f}"
        elif status.get("latest_train_loss") is not None:
            metric = f"train={status['latest_train_loss']:.4f}"
        run_id = status.get("run_id", "?")
        state = status.get("state", "?")
        phase = status.get("phase", "?")
        step = status.get("step", "-")
        backend = status.get("backend", "-")
        age_display = f"{age:.0f}" if age is not None else "-"
        stale_flag = (
            " stale"
            if age is not None and age > stale_seconds and state in ACTIVE_STATES
            else ""
        )
        lines.append(
            f"{run_id:<18} {state:<18} {phase:<12} {age_display:<8} "
            f"{step!s:<8} {metric:<18} {backend:<8}{stale_flag}"
        )
        last_line = status.get("last_output_line")
        if last_line:
            lines.append(f"    {last_line[:120]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fleet-wide monitor (probes all nodes)
# ---------------------------------------------------------------------------

def render_monitor(nodes: list[NodeRecord]) -> str:
    """Probe every node and render a fleet-wide status table."""
    lines = [
        f"{'Node':<18} {'Ready':<14} {'Run':<28} {'State':<18} "
        f"{'Step':<8} {'Metric':<18}"
    ]
    lines.append("-" * 120)
    for node in nodes:
        status = probe_node_status(node)
        metric = "-"
        if status.get("latest_val_loss") is not None:
            metric = f"val_loss={status['latest_val_loss']:.4f}"
        elif status.get("latest_train_loss") is not None:
            metric = f"train={status['latest_train_loss']:.4f}"
        lines.append(
            f"{node['name']:<18} {ready_state(node):<14} "
            f"{status.get('run_id', '-'):<28} "
            f"{status.get('state', '-'):<18} "
            f"{str(status.get('step', '-')):<8} {metric:<18}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render node inventory table
# ---------------------------------------------------------------------------

def render_nodes(nodes: list[NodeRecord]) -> str:
    """Pretty-print the node inventory."""
    lines = [
        f"{'Name':<18} {'GPU':<12} {'Host':<18} {'Port':<7} "
        f"{'State':<14} {'Git':<10} {'$hr':<6}"
    ]
    lines.append("-" * 90)
    for node in nodes:
        git_sha = (node.get("git_sha") or "-")[:8]
        lines.append(
            f"{node['name']:<18} {node.get('gpu', '-'):<12} "
            f"{node.get('ssh_host', '-'):<18} "
            f"{str(node.get('ssh_port', '-')):<7} "
            f"{ready_state(node):<14} {git_sha:<10} "
            f"{str(node.get('cost_per_hr', '-')):<6}"
        )
    return "\n".join(lines)
