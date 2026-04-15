"""Node inventory CRUD, merge, health classification, and state tracking."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from crucible.core.errors import FleetError
from crucible.core.io import atomic_write_json
from crucible.core.log import utc_now_iso
from crucible.core.types import NodeRecord
from crucible.fleet.sync import ssh_ok

# ---------------------------------------------------------------------------
# Module-level lock for thread-safe node file access
# ---------------------------------------------------------------------------
NODES_LOCK = threading.Lock()

# States that indicate a node is definitively bad and should not be used.
BAD_API_STATES: set[str] = frozenset(
    {"paused", "stopped", "terminated", "failed", "cancelled", "exited", "lost"}
)

# Intentionally paused states — not usable but not broken, bootstrap flags preserved.
PAUSED_STATES: set[str] = frozenset({"stopped", "exited"})

TERMINAL_RESULT_STATUSES: set[str] = frozenset(
    {"completed", "partial_recoverable", "failed", "timeout", "killed", "early_stopped"}
)


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_nodes(path: Path) -> list[NodeRecord]:
    """Load nodes from a JSON file.  Raises FleetError if the file is missing
    or malformed."""
    if not path.exists():
        raise FleetError(f"Nodes file not found: {path}")
    nodes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(nodes, list):
        raise FleetError(f"Nodes file must contain a JSON list: {path}")
    return nodes


def load_nodes_if_exists(path: Path) -> list[NodeRecord]:
    """Like :func:`load_nodes` but returns ``[]`` when the file is absent."""
    if not path.exists():
        return []
    return load_nodes(path)


def save_nodes(path: Path, nodes: list[NodeRecord]) -> None:
    """Atomically persist the nodes list to *path*."""
    atomic_write_json(path, nodes)


def save_nodes_threadsafe(path: Path, nodes: list[NodeRecord]) -> None:
    """Thread-safe wrapper around :func:`save_nodes`."""
    with NODES_LOCK:
        save_nodes(path, nodes)


def load_nodes_snapshot(path: Path) -> list[NodeRecord]:
    """Thread-safe read of the current on-disk node list."""
    with NODES_LOCK:
        return load_nodes_if_exists(path)


# ---------------------------------------------------------------------------
# Upsert / Merge
# ---------------------------------------------------------------------------

def upsert_node_record(path: Path, node: NodeRecord) -> list[NodeRecord]:
    """Insert or update a single node in the on-disk list.

    Matching is by ``node_id`` (or ``pod_id`` for backward compat) then ``name``.
    """
    with NODES_LOCK:
        nodes = load_nodes_if_exists(path)
        updated = False
        node_id = node.get("node_id") or node.get("pod_id")
        for idx, existing in enumerate(nodes):
            existing_id = existing.get("node_id") or existing.get("pod_id")
            if (existing_id and existing_id == node_id) or existing.get("name") == node.get("name"):
                nodes[idx] = node
                updated = True
                break
        if not updated:
            nodes.append(node)
        save_nodes(path, nodes)
        return nodes


def merge_node_record(existing: NodeRecord | None, incoming: NodeRecord) -> NodeRecord:
    """Merge an *incoming* node record on top of an *existing* one.

    Preserves local bootstrap flags when API data does not supersede them.
    """
    if existing is None:
        return dict(incoming)
    merged = dict(existing)
    merged.update(incoming)
    merged["env_ready"] = bool(existing.get("env_ready") or incoming.get("env_ready"))
    merged["dataset_ready"] = bool(existing.get("dataset_ready") or incoming.get("dataset_ready"))
    merged["replacement"] = bool(existing.get("replacement") or incoming.get("replacement"))
    merged["git_sha"] = incoming.get("git_sha") or existing.get("git_sha")
    merged["last_seen_at"] = incoming.get("last_seen_at") or existing.get("last_seen_at")
    merged["ssh_host"] = incoming.get("ssh_host") or existing.get("ssh_host") or ""
    merged["ssh_port"] = incoming.get("ssh_port") or existing.get("ssh_port") or 22
    merged["project"] = incoming.get("project") or existing.get("project")
    merged["workspace_path"] = incoming.get("workspace_path") or existing.get("workspace_path")
    merged["env_source"] = incoming.get("env_source") or existing.get("env_source")
    merged["python_bin"] = incoming.get("python_bin") or existing.get("python_bin")
    api_state = str(merged.get("api_state") or "").lower()
    existing_state = str(existing.get("state") or "").lower()
    # Intentionally paused pods: keep "stopped" state, preserve bootstrap flags.
    # Skip if the existing local state is "starting" (pod is actively resuming).
    if api_state in PAUSED_STATES and existing_state not in {"starting"}:
        merged["state"] = "stopped"
        return merged
    if existing_state in {"ready", "boot_failed", "unreachable", "ssh_timeout"} and api_state not in BAD_API_STATES:
        merged["state"] = existing_state
    if merged["env_ready"] and merged["dataset_ready"] and merged["git_sha"] and api_state not in BAD_API_STATES:
        merged["state"] = "ready"
    return merged


def merge_node_snapshots(
    existing_nodes: list[NodeRecord],
    incoming_nodes: list[NodeRecord],
) -> list[NodeRecord]:
    """Three-way merge: incoming records update existing ones; orphans are kept."""
    existing_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for node in existing_nodes:
        key = (node.get("node_id") or node.get("pod_id"), node.get("name"))
        existing_by_key[key] = node

    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for node in incoming_nodes:
        key = (node.get("node_id") or node.get("pod_id"), node.get("name"))
        merged.append(merge_node_record(existing_by_key.get(key), node))
        seen.add(key)
    for node in existing_nodes:
        key = (node.get("node_id") or node.get("pod_id"), node.get("name"))
        if key not in seen:
            merged.append(node)
    return merged


# ---------------------------------------------------------------------------
# Ready state / health classification
# ---------------------------------------------------------------------------

def ready_state(node: NodeRecord) -> str:
    """Determine if a node is fully bootstrapped and usable."""
    if not node.get("env_ready"):
        return "env_missing"
    if not node.get("dataset_ready"):
        return "dataset_missing"
    if not node.get("git_sha"):
        return "unsynced"
    return node.get("state", "ready")


def classify_health(node: NodeRecord) -> str:
    """Probe a node and return a single health label (``ready``, ``unreachable``, etc.)."""
    api_state = str(node.get("api_state") or "").lower()
    state = str(node.get("state") or "").lower()
    # Intentionally paused — not broken, just idle
    if api_state in PAUSED_STATES or state in PAUSED_STATES:
        return "stopped"
    if api_state in BAD_API_STATES or state in BAD_API_STATES:
        return api_state or state
    if state in {"creating", "new", "bootstrapping"}:
        return state
    if not node.get("env_ready") or not node.get("dataset_ready") or not node.get("git_sha"):
        return ready_state(node)
    if not ssh_ok(node):
        return "unreachable"
    return "ready"


def count_bootstrapped_ready(nodes: list[NodeRecord]) -> int:
    """Return the number of fully bootstrapped & ready nodes."""
    return sum(1 for n in nodes if ready_state(n) == "ready")


def next_node_index(nodes: list[NodeRecord], name_prefix: str) -> int:
    """Find the next ordinal index for a new node with the given prefix."""
    prefix = f"{name_prefix}-"
    indexes: list[int] = []
    for node in nodes:
        name = node.get("name", "")
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.isdigit():
            indexes.append(int(suffix))
    return (max(indexes) if indexes else 0) + 1


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_nodes(nodes: list[NodeRecord]) -> dict[str, int]:
    """Return a dict of aggregate counts for the fleet status display."""
    healthy_states = {"ready"}
    unhealthy_states = (BAD_API_STATES - PAUSED_STATES) | {"unreachable", "ssh_timeout", "boot_failed"}
    return {
        "nodes_total": len(nodes),
        "nodes_ready": sum(1 for n in nodes if str(n.get("state", "")).lower() in healthy_states),
        "nodes_bootstrapped": sum(
            1 for n in nodes if n.get("env_ready") and n.get("dataset_ready") and n.get("git_sha")
        ),
        "nodes_failed": sum(1 for n in nodes if str(n.get("state", "")).lower() in unhealthy_states),
        "nodes_healthy": sum(1 for n in nodes if str(n.get("state", "")).lower() in healthy_states),
        "nodes_unhealthy": sum(1 for n in nodes if str(n.get("state", "")).lower() in unhealthy_states),
        "nodes_stopped": sum(1 for n in nodes if str(n.get("state", "")).lower() in PAUSED_STATES),
        "nodes_replaced": sum(1 for n in nodes if n.get("replacement")),
        "nodes_bootstrapping": sum(
            1 for n in nodes if str(n.get("state", "")).lower() in {"creating", "new", "bootstrapping"}
        ),
    }
