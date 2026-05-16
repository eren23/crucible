"""Per-session cost tracking for autonomous research loops (Phase 1.8).

Plumbs a simple wall-clock × declared-hourly-rate spend model. Cost data
comes from active fleet nodes' ``cost_per_hr`` field (already populated
by ``fleet/providers/runpod.py`` from RunPod's ``adjustedCostPerHr`` /
``costPerHr`` GraphQL fields). A session's spend is:

    spend = (now - session_started_at) × sum(active_nodes.cost_per_hr)

Two fallback paths:
- **No nodes.json (or no active pods):** rate is 0, spend is 0. The
  budget cap is effectively unenforceable because no pods are running
  and therefore no money is being spent. This is correct for dry-run
  development and for sessions that exit before provisioning.
- **Active nodes that lack a positive cost_per_hr** (e.g., SSH provider
  nodes which never populate the field): the per-pod rate falls back
  to ``DEFAULT_FALLBACK_HOURLY_USD`` so the cap is still enforceable
  for self-hosted setups. A one-time log_warn surfaces the fallback.

The model is intentionally coarse — autonomous sessions in the same
project share the same fleet, so per-session attribution can't be
exact. The MVP semantic is "wall-clock since this session started ×
current pod rate" — close enough to enforce a budget cap, transparent
when surfaced in session state.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.log import log_warn

# Per-pod fallback rate when no cost_per_hr is available on the fleet
# nodes. Chosen to be conservative — small enough not to blow a $5 cap
# in 10 minutes, large enough to actually trip on hour-scale waste.
DEFAULT_FALLBACK_HOURLY_USD = 0.50


def _load_nodes(config: ProjectConfig) -> list[dict[str, Any]]:
    """Read the project's nodes.json. Returns empty list on any error."""
    nodes_path = Path(config.project_root) / config.nodes_file
    if not nodes_path.exists():
        return []
    try:
        raw = json.loads(nodes_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(raw, list):
        return [n for n in raw if isinstance(n, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("nodes"), list):
        return [n for n in raw["nodes"] if isinstance(n, dict)]
    return []


def active_pod_hourly_rate(config: ProjectConfig) -> float:
    """Sum the ``cost_per_hr`` of all active (non-terminal) fleet nodes.

    "Active" means not in a terminal state (``destroyed``, ``stopped``,
    ``terminated``) — we count pods that could be consuming money right
    now. If no nodes carry a ``cost_per_hr`` field, falls back to a
    conservative per-pod default.
    """
    terminal_states = {"destroyed", "stopped", "terminated", "failed"}
    nodes = _load_nodes(config)
    if not nodes:
        return 0.0

    rates: list[float] = []
    fallback_used = False
    for node in nodes:
        state = (node.get("state") or "").lower()
        if state in terminal_states:
            continue
        rate = node.get("cost_per_hr")
        try:
            rate = float(rate) if rate is not None else None
        except (TypeError, ValueError):
            rate = None
        if rate is None or rate <= 0:
            rate = DEFAULT_FALLBACK_HOURLY_USD
            fallback_used = True
        rates.append(rate)

    if not rates:
        return 0.0
    if fallback_used:
        log_warn(
            "cost_tracker: some active nodes lack a positive cost_per_hr; "
            f"using ${DEFAULT_FALLBACK_HOURLY_USD:.2f}/hr fallback per affected pod."
        )
    return sum(rates)


def compute_session_spend(
    config: ProjectConfig,
    session_started_at: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Estimate per-session spend so far.

    Returns ``{spend_usd, hours_elapsed, hourly_rate, active_pods}``. All
    are best-effort estimates; downstream callers compare ``spend_usd``
    to the session's ``budget_usd`` cap.
    """
    started = _parse_iso(session_started_at)
    if started is None:
        return {
            "spend_usd": 0.0,
            "hours_elapsed": 0.0,
            "hourly_rate": 0.0,
            "active_pods": 0,
            "error": f"could not parse session_started_at={session_started_at!r}",
        }
    current = now or datetime.now(timezone.utc)
    hours = max(0.0, (current - started).total_seconds() / 3600.0)
    rate = active_pod_hourly_rate(config)
    active_pods = sum(
        1
        for n in _load_nodes(config)
        if (n.get("state") or "").lower()
        not in {"destroyed", "stopped", "terminated", "failed"}
    )
    return {
        "spend_usd": round(hours * rate, 4),
        "hours_elapsed": round(hours, 4),
        "hourly_rate": round(rate, 4),
        "active_pods": active_pods,
    }


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # utc_now_iso() emits e.g. "2026-05-15T07:51:23.456789+00:00"
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_FALLBACK_HOURLY_USD",
    "active_pod_hourly_rate",
    "compute_session_spend",
]
