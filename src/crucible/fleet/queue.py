"""Experiment queue: enqueue, dispatch, reconcile, save, load."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from crucible.core.io import read_jsonl, write_jsonl
from crucible.core.log import utc_now_iso, utc_stamp


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def make_run_id(name: str) -> str:
    """Create a unique run identifier from an experiment name."""
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name)[:32]
    return f"{utc_stamp()}-{slug}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Queue persistence
# ---------------------------------------------------------------------------

def load_queue(path: Path) -> list[dict[str, Any]]:
    """Read the fleet queue from a JSONL file."""
    return read_jsonl(path)


def save_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the entire fleet queue atomically."""
    write_jsonl(path, rows)


def reset_queue(path: Path) -> None:
    """Clear the fleet queue."""
    save_queue(path, [])


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

def enqueue_experiments(
    queue_path: Path,
    experiments: list[dict[str, Any]],
    *,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Append new experiments to the queue, skipping duplicates.

    Duplicate detection uses ``(experiment_name, tier)`` as the key.
    When *limit* > 0, at most *limit* experiments are added.

    Returns the list of newly added queue items.
    """
    existing = load_queue(queue_path)
    existing_keys = {(row["experiment_name"], row["tier"]) for row in existing}
    added: list[dict[str, Any]] = []
    for exp in experiments:
        if limit > 0 and len(added) >= limit:
            break
        key = (exp["name"], exp["tier"])
        if key in existing_keys:
            continue
        added.append(
            {
                "experiment_name": exp["name"],
                "run_id": make_run_id(exp["name"]),
                "tier": exp["tier"],
                "backend": exp["backend"],
                "config": exp["config"],
                "tags": exp.get("tags", []),
                "priority": exp.get("priority", 0),
                "wave": exp.get("wave"),
                "assigned_node": None,
                "lease_state": "queued",
                "attempt": 0,
                "created_at": utc_now_iso(),
                "started_at": None,
                "ended_at": None,
                "result_status": None,
            }
        )
    if added:
        save_queue(queue_path, existing + added)
    return added


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

def reconcile_queue_with_results(
    rows: list[dict[str, Any]],
    result_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match queue items with collected results and update lease states."""
    updated: list[dict[str, Any]] = []
    for item in rows:
        result = result_index.get(item["run_id"])
        if result is not None:
            item["result_status"] = result.get("status")
            item["ended_at"] = item.get("ended_at") or utc_now_iso()
            item["lease_state"] = "completed" if result.get("status") == "completed" else "finished"
        updated.append(item)
    return updated


# ---------------------------------------------------------------------------
# Queue summary helpers
# ---------------------------------------------------------------------------

def summarize_queue(rows: list[dict[str, Any]], *, wave_name: str | None = None) -> dict[str, int]:
    """Return aggregate counts for the queue (optionally scoped to a wave)."""
    scoped = [row for row in rows if wave_name is None or row.get("wave") == wave_name]
    return {
        "queue_total": len(scoped),
        "queue_running": sum(1 for row in scoped if row.get("lease_state") == "running"),
        "queue_queued": sum(1 for row in scoped if row.get("lease_state") == "queued"),
        "queue_finished": sum(1 for row in scoped if row.get("lease_state") in {"completed", "finished"}),
        "queue_completed": sum(
            1 for row in scoped
            if row.get("lease_state") == "completed" or row.get("result_status") == "completed"
        ),
        "retryable_runs": sum(1 for row in scoped if row.get("lease_state") == "retryable"),
        "failed_retry_budget": sum(1 for row in scoped if row.get("result_status") == "failed_retry_budget"),
    }


def summarize_idle_capacity(
    nodes: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    *,
    wave_name: str | None = None,
) -> dict[str, int]:
    """Count ready nodes with no active assignment."""
    from crucible.fleet.inventory import ready_state

    active_assignments = {
        row.get("assigned_node") or row.get("assigned_pod")
        for row in queue
        if (wave_name is None or row.get("wave") == wave_name)
        and row.get("lease_state") == "running"
    }
    ready_idle = [
        node for node in nodes
        if ready_state(node) == "ready" and node.get("name") not in active_assignments
    ]
    return {"nodes_ready_idle": len(ready_idle)}


def wave_rows(queue: list[dict[str, Any]], wave_name: str) -> list[dict[str, Any]]:
    """Filter queue to items belonging to a specific wave."""
    return [row for row in queue if row.get("wave") == wave_name]


def wave_result_rows(
    queue: list[dict[str, Any]],
    result_index: dict[str, dict[str, Any]],
    wave_name: str,
) -> list[dict[str, Any]]:
    """Collect actual result dicts for completed items in a wave."""
    rows = []
    for item in wave_rows(queue, wave_name):
        result = result_index.get(item["run_id"])
        if result is not None:
            rows.append(result)
    return rows


def results_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index results by their ``id`` field."""
    return {row["id"]: row for row in rows if "id" in row}


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def load_wave_spec(path: Path, *, default_backend: str = "torch") -> list[dict[str, Any]]:
    """Load and validate a JSON wave/spec file.

    Each entry must have ``name`` (str) and ``config`` (dict).
    Optional fields: ``tags``, ``priority``, ``tier``, ``backend``, ``wave``.
    """
    if not path.exists():
        raise SystemExit(f"Spec file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Spec file must contain a JSON array: {path}")
    normalized: list[dict[str, Any]] = []
    inferred_wave = path.stem
    for idx, entry in enumerate(payload, 1):
        if not isinstance(entry, dict):
            raise SystemExit(f"Spec entry #{idx} in {path} must be a JSON object.")
        if not entry.get("name") or not isinstance(entry.get("name"), str):
            raise SystemExit(f"Spec entry #{idx} in {path} is missing string field 'name'.")
        config = entry.get("config")
        if not isinstance(config, dict):
            raise SystemExit(f"Spec entry #{idx} in {path} is missing object field 'config'.")
        tags = entry.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise SystemExit(f"Spec entry #{idx} in {path} has invalid 'tags'; expected string array.")
        normalized.append(
            {
                "name": entry["name"],
                "config": config,
                "tags": tags,
                "priority": int(entry.get("priority", 0)),
                "tier": entry.get("tier", "proxy"),
                "backend": entry.get("backend", default_backend),
                "wave": entry.get("wave", inferred_wave),
            }
        )
    return normalized


def prepare_wave_experiments(
    experiments: list[dict[str, Any]],
    *,
    extra_tags: list[str],
) -> list[dict[str, Any]]:
    """Clone experiments with additional tags merged in (preserving order)."""
    prepared: list[dict[str, Any]] = []
    for exp in experiments:
        tags = list(dict.fromkeys([*exp.get("tags", []), *extra_tags]))
        prepared.append({**exp, "tags": tags})
    return prepared
