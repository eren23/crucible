"""Append-only per-iteration evolution log for harness optimization.

Each record captures what happened in a single propose→validate→benchmark
→frontier cycle. The log lives at
``{tree_dir}/evolution_log.jsonl`` and is reconstructable with :func:`read_log`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.io import append_jsonl, read_jsonl
from crucible.core.log import utc_now_iso


def log_path(tree_dir: Path) -> Path:
    return Path(tree_dir) / "evolution_log.jsonl"


def append_iteration(
    tree_dir: Path,
    *,
    iteration: int,
    proposed: int,
    validated: int,
    benchmarked: int = 0,
    frontier_summary: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a single iteration record; returns the written record."""
    record: dict[str, Any] = {
        "iteration": iteration,
        "timestamp": utc_now_iso(),
        "proposed": proposed,
        "validated": validated,
        "benchmarked": benchmarked,
        "frontier_summary": frontier_summary or {},
        "cost": cost or {},
        "notes": notes,
    }
    if extra:
        record.update(extra)
    append_jsonl(log_path(tree_dir), record)
    return record


def read_log(tree_dir: Path) -> list[dict[str, Any]]:
    """Return all iteration records, oldest first. Empty list when missing."""
    return read_jsonl(log_path(tree_dir))


def last_iteration(tree_dir: Path) -> int:
    """Return the largest ``iteration`` seen (0 when log is empty).

    Used on restart to resume numbering without re-parsing every record in
    the consumer; we keep this as a small helper so callers don't duplicate
    the max-scan.
    """
    records = read_log(tree_dir)
    if not records:
        return 0
    return max(int(r.get("iteration", 0)) for r in records)
