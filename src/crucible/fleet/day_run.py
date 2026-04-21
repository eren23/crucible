"""Day run directory management, events, summaries, leaderboard."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from crucible.core.io import atomic_write_json, _json_ready
from crucible.core.log import utc_now_iso, utc_stamp
from crucible.core.types import ExperimentResult, JsonDict, JsonValue, NodeRecord, QueueItem

# Thread lock for event file appends
EVENTS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Day run directory
# ---------------------------------------------------------------------------

def create_day_run_dir(base_dir: Path) -> Path:
    """Create a timestamped day-run directory under *base_dir*."""
    path = base_dir / utc_stamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def day_tag(day_dir: Path) -> str:
    """Derive a tag string from a day-run directory."""
    return f"dayrun-{day_dir.name}"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def append_event(day_dir: Path, event_type: str, **fields: JsonValue) -> None:
    """Append a single event record to the day's events.jsonl."""
    event = {"ts": utc_now_iso(), "event": event_type, **_json_ready(fields)}
    path = day_dir / "events.jsonl"
    with EVENTS_LOCK:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(existing + json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_day_summary(day_dir: Path, summary: JsonDict) -> None:
    """Write (or overwrite) the day summary JSON file."""
    atomic_write_json(day_dir / "summary.json", summary)


def update_day_summary(day_dir: Path, summary: JsonDict, **updates: JsonValue) -> None:
    """Merge *updates* into *summary*, set ``last_updated_at``, and persist."""
    summary.update(updates)
    summary["last_updated_at"] = utc_now_iso()
    write_day_summary(day_dir, summary)


# ---------------------------------------------------------------------------
# Progress recording
# ---------------------------------------------------------------------------

def record_day_progress(
    day_dir: Path,
    summary: dict[str, Any],
    *,
    phase: str | None = None,
    current_wave: str | None = None,
    last_event: str | None = None,
    nodes: list[NodeRecord] | None = None,
    queue: list[QueueItem] | None = None,
    wave_name: str | None = None,
) -> None:
    """Update the day summary with the latest fleet progress."""
    from crucible.fleet.inventory import summarize_nodes, count_bootstrapped_ready
    from crucible.fleet.queue import summarize_queue, summarize_idle_capacity

    updates: dict[str, Any] = {}
    if phase is not None:
        updates["phase"] = phase
    if current_wave is not None:
        updates["current_wave"] = current_wave
    if last_event is not None:
        updates["last_event"] = last_event
    if nodes is not None:
        updates.update(summarize_nodes(nodes))
        updates["bootstrap_ready_count"] = count_bootstrapped_ready(nodes)
    if queue is not None:
        updates.update(summarize_queue(queue, wave_name=wave_name))
        if nodes is not None:
            updates.update(summarize_idle_capacity(nodes, queue, wave_name=wave_name))
    update_day_summary(day_dir, summary, **updates)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def write_day_leaderboard(
    day_dir: Path,
    rows: list[ExperimentResult],
    metric_key: str = "val_loss",
) -> None:
    """Generate a markdown leaderboard ranked by *metric_key*."""
    ranked = [
        row
        for row in rows
        if row.get("status") == "completed" and row.get("result") and metric_key in row["result"]
    ]
    ranked.sort(key=lambda row: row["result"][metric_key])
    lines = [
        "# Day Leaderboard", "",
        f"| Rank | Name | {metric_key} | Steps |",
        "|---:|---|---:|---:|",
    ]
    for idx, row in enumerate(ranked, 1):
        res = row["result"]
        lines.append(f"| {idx} | {row['name']} | {res[metric_key]:.4f} | {res.get('steps_completed', 0)} |")
    (day_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def leaderboard_snippet(
    rows: list[dict[str, Any]],
    *,
    limit: int = 3,
    metric_key: str = "val_loss",
) -> list[str]:
    """Return the top *limit* results as short summary strings."""
    ranked = [
        row
        for row in rows
        if row.get("status") == "completed" and row.get("result") and metric_key in row["result"]
    ]
    ranked.sort(key=lambda row: row["result"][metric_key])
    out: list[str] = []
    for idx, row in enumerate(ranked[:limit], 1):
        out.append(f"{idx}. {row['name']} {metric_key}={row['result'][metric_key]:.4f}")
    return out


# ---------------------------------------------------------------------------
# Day status rendering
# ---------------------------------------------------------------------------

def remaining_waves(summary: dict[str, Any]) -> list[str]:
    """Return the list of waves that haven't completed yet."""
    order = summary.get("wave_order", [])
    completed = {row["name"] for row in summary.get("waves", [])}
    current = summary.get("current_wave")
    remaining: list[str] = []
    for wave in order:
        if wave in completed:
            continue
        if current == wave:
            remaining.append(f"{wave} tail")
        else:
            remaining.append(wave)
    return remaining


def render_day_status(summary: dict[str, Any]) -> str:
    """Render a human-readable day run status string."""
    lines = [
        f"Day: {summary.get('day_tag', '-')}",
        f"Phase: {summary.get('phase', '-')}",
        f"Wave: {summary.get('current_wave') or '-'}",
        f"Launch started: {'yes' if summary.get('wave1_started') else 'no'}",
        (
            f"Nodes: healthy {summary.get('nodes_healthy', summary.get('nodes_ready', 0))}"
            f"/{summary.get('nodes_total', 0)}, "
            f"bootstrapped {summary.get('nodes_bootstrapped', 0)}"
            f"/{summary.get('nodes_total', 0)}, "
            f"replaced {summary.get('nodes_replaced', 0)}"
        ),
        (
            f"Idle ready nodes: {summary.get('nodes_ready_idle', 0)} "
            f"| queued runs: {summary.get('queue_queued', 0)}"
        ),
        (
            f"Bootstrap: ready {summary.get('bootstrap_ready_count', summary.get('nodes_bootstrapped', 0))}, "
            f"min start {summary.get('min_ready_to_start', 0)}, "
            f"replacements used {summary.get('replacements_used', 0)}"
        ),
        (
            f"Runs: completed {summary.get('queue_completed', 0)}, "
            f"running {summary.get('queue_running', 0)}, "
            f"retryable {summary.get('retryable_runs', 0)}, "
            f"finished {summary.get('queue_finished', 0)}/{summary.get('queue_total', 0)}"
        ),
        f"Left today: {', '.join(remaining_waves(summary)) or 'none'}",
        f"Summary: {summary.get('summary_path', '-')}",
    ]
    return "\n".join(lines)
