"""Node status probing, render functions, live monitoring."""
from __future__ import annotations

import json
import shlex
from typing import Any

from crucible.core.types import NodeRecord
from crucible.fleet.inventory import ready_state
from crucible.fleet.sync import remote_exec


# ---------------------------------------------------------------------------
# Status probing
# ---------------------------------------------------------------------------


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
