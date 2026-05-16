"""Tool router — state-aware MCP tool recommendation.

Inspects the current project state (fleet, queue, leaderboard, research
state, active sessions) and returns the recommended next MCP tool with
rationale and a list of alternatives.

This is Phase 2.1 — the foundation that makes the 200+ MCP surface
navigable. Existing REQUIRES/RETURNS/NEXT prose in tool descriptions is
not machine-readable; instead the router hard-codes a decision graph
based on the canonical lifecycle:

    provision_nodes → bootstrap_nodes → design_enqueue_batch →
    dispatch_experiments → collect_results → get_leaderboard →
    research_request_prompt(stage=reflection) → repeat

with branches for active autonomous sessions, orphan pods, empty
hypotheses, and budget exhaustion.

Pure heuristic — no LLM. Callers are orchestrators (Claude Code, Codex,
human via CLI) that need a hint without parsing 422 prose blocks.
"""
from __future__ import annotations

from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.log import log_warn


# Reasons map to action-rationale tuples. Strings are user-visible.
_R_NO_PODS = (
    "provision_nodes",
    "No pods provisioned for this project. Provision nodes before bootstrapping.",
)
_R_PODS_NOT_REFRESHED = (
    "fleet_refresh",
    "Pods exist but SSH endpoints are stale. Refresh to pick up provider state.",
)
_R_PODS_NOT_BOOTSTRAPPED = (
    "bootstrap_nodes",
    "Pods are provisioned but not bootstrapped (code/env not synced).",
)
_R_ORPHANS_PRESENT = (
    "cleanup_orphans",
    "Provider has pods tagged for this project but missing from local inventory.",
)
_R_NO_HYPOTHESES = (
    "research_request_prompt",
    "No hypotheses recorded yet. Request a hypothesis prompt (stage='hypothesis').",
)
_R_HYPOTHESES_NO_BATCH = (
    "design_batch_from_hypotheses",
    "Pending hypotheses exist but no batch has been built yet.",
)
_R_QUEUE_EMPTY_HAVE_HYPS = (
    "design_enqueue_batch",
    "Queue is empty but pending hypotheses are ready to be enqueued.",
)
_R_DISPATCH = (
    "dispatch_experiments",
    "Experiments queued and at least one pod is idle.",
)
_R_RUNNING = (
    "get_fleet_status",
    "Experiments are running. Poll fleet status until they complete.",
)
_R_COLLECT = (
    "collect_results",
    "Experiments finished but results have not been rsynced from pods.",
)
_R_LEADERBOARD_FRESH = (
    "get_leaderboard",
    "New results are in. Inspect the leaderboard before forming the next hypothesis.",
)
_R_REFLECT = (
    "research_request_prompt",
    "Leaderboard has fresh entries. Request a reflection prompt (stage='reflection').",
)
_R_BRIEFING_DEFAULT = (
    "get_research_briefing",
    "No clear next step from state alone — pull a research briefing for orientation.",
)


def _load_nodes(config: ProjectConfig) -> list[dict[str, Any]]:
    try:
        from crucible.fleet.inventory import load_nodes
        return load_nodes(config.project_root / config.nodes_file)
    except Exception as exc:
        log_warn(f"tool_router: load_nodes failed: {exc}")
        return []


def _load_queue(config: ProjectConfig) -> list[dict[str, Any]]:
    try:
        from crucible.fleet.queue import load_queue
        return load_queue(config.project_root / "fleet_queue.jsonl")
    except Exception as exc:
        log_warn(f"tool_router: load_queue failed: {exc}")
        return []


def _load_completed_count(config: ProjectConfig) -> int:
    try:
        from crucible.analysis.results import completed_results
        return len(completed_results(config))
    except Exception as exc:
        log_warn(f"tool_router: completed_results failed: {exc}")
        return 0


def _load_research_state(config: ProjectConfig) -> dict[str, Any]:
    state_path = config.project_root / config.research_state_file
    if not state_path.exists():
        return {"available": False, "hypotheses": [], "pending_count": 0,
                "history_count": 0, "budget_remaining": 0.0}
    try:
        from crucible.researcher.state import ResearchState
        state = ResearchState(state_path)
        hyps = state.hypotheses or []
        pending = [h for h in hyps if h.get("status") == "pending"]
        return {
            "available": True,
            "hypotheses": hyps,
            "pending_count": len(pending),
            "history_count": len(state.history or []),
            "budget_remaining": float(state.budget_remaining or 0.0),
        }
    except Exception as exc:
        log_warn(f"tool_router: ResearchState load failed: {exc}")
        return {"available": False, "hypotheses": [], "pending_count": 0,
                "history_count": 0, "budget_remaining": 0.0}


def _find_active_session(config: ProjectConfig) -> dict[str, Any] | None:
    """Return active session info across all three session types, or None.

    Checks autonomous (research), tree, and harness session dirs for a
    non-terminal yaml. Returns the most-recently-updated active one.
    """
    candidates: list[tuple[str, Any]] = []
    try:
        from crucible.researcher.autonomous_session import AutonomousSession
        active = AutonomousSession._find_active_yamls(config)
        for sid, _, data in active:
            candidates.append(("autonomous", (sid, data)))
    except Exception as exc:
        log_warn(f"tool_router: autonomous session scan failed: {exc}")
    try:
        from crucible.researcher.tree_autonomous_session import TreeAutonomousSession
        active = TreeAutonomousSession._find_active_yamls(config)
        for sid, _, data in active:
            candidates.append(("tree", (sid, data)))
    except Exception as exc:
        log_warn(f"tool_router: tree session scan failed: {exc}")
    try:
        from crucible.researcher.harness_autonomous_session import HarnessAutonomousSession
        active = HarnessAutonomousSession._find_active_yamls(config)
        for sid, _, data in active:
            candidates.append(("harness", (sid, data)))
    except Exception as exc:
        log_warn(f"tool_router: harness session scan failed: {exc}")

    if not candidates:
        return None
    # _find_active_yamls returns descending by updated_at; first across all
    # is the most recent.
    candidates.sort(key=lambda c: c[1][1].get("updated_at", ""), reverse=True)
    kind, (sid, data) = candidates[0]
    return {
        "kind": kind,
        "session_id": sid,
        "stage": data.get("stage"),
        "status": data.get("status"),
    }


def _node_state_summary(nodes: list[dict[str, Any]]) -> dict[str, int]:
    """Count nodes by ready / unready / dead."""
    summary = {"total": len(nodes), "ready": 0, "unbootstrapped": 0,
               "no_ssh": 0, "dead": 0}
    for n in nodes:
        state = (n.get("state") or "").lower()
        if state in {"destroyed", "terminated", "failed"}:
            summary["dead"] += 1
            continue
        if not n.get("ssh_host"):
            summary["no_ssh"] += 1
            continue
        if not n.get("env_ready"):
            summary["unbootstrapped"] += 1
            continue
        summary["ready"] += 1
    return summary


def _queue_state_summary(queue: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total": len(queue), "queued": 0, "running": 0,
               "finished": 0, "failed": 0}
    for row in queue:
        status = (row.get("status") or "queued").lower()
        if status == "queued":
            summary["queued"] += 1
        elif status == "running":
            summary["running"] += 1
        elif status in {"completed", "finished", "success"}:
            summary["finished"] += 1
        elif status in {"failed", "error", "canceled"}:
            summary["failed"] += 1
    return summary


def recommend_next_action(config: ProjectConfig) -> dict[str, Any]:
    """Inspect project state and return a recommended next MCP tool.

    Return shape:
        {
          "recommended_tool": str,
          "rationale": str,
          "alternatives": [{"tool": str, "rationale": str}, ...],
          "state": {
            "nodes": {"total": int, "ready": int, "unbootstrapped": int, ...},
            "queue": {"total": int, "queued": int, "running": int, ...},
            "completed_experiments": int,
            "hypotheses_pending": int,
            "active_session": {...} | None,
            "orphans_present": bool,
          },
        }

    The decision tree, top to bottom:
        1. Active autonomous session → continue/status that session
        2. Orphan pods present → cleanup_orphans
        3. No pods → provision_nodes
        4. Pods missing SSH → fleet_refresh
        5. Pods present, none bootstrapped → bootstrap_nodes
        6. Experiments running → poll fleet
        7. Experiments queued + idle pods → dispatch
        8. Queue empty + hypotheses pending → design_batch/enqueue
        9. Queue empty + no hypotheses + completed experiments → reflect
        10. Queue empty + no hypotheses + no experiments → request hypothesis prompt
        11. Default → briefing
    """
    nodes = _load_nodes(config)
    queue = _load_queue(config)
    completed_count = _load_completed_count(config)
    rs = _load_research_state(config)
    active = _find_active_session(config)

    # Cheap orphan probe — best-effort; many setups don't have a provider
    # client wired up. Only flag if we can ask without exploding.
    orphans_present = False
    try:
        from crucible.fleet.manager import FleetManager
        fm = FleetManager(config)
        info = fm.cleanup_orphans(destroy=False, include_legacy=False)
        orphans_present = bool(info.get("tagged_orphans"))
    except Exception:
        # Provider not configured / not implemented — silent.
        pass

    nodes_sum = _node_state_summary(nodes)
    queue_sum = _queue_state_summary(queue)

    state = {
        "nodes": nodes_sum,
        "queue": queue_sum,
        "completed_experiments": completed_count,
        "hypotheses_pending": rs["pending_count"],
        "hypotheses_total": len(rs["hypotheses"]),
        "history_count": rs["history_count"],
        "active_session": active,
        "orphans_present": orphans_present,
    }

    primary: tuple[str, str]
    alternatives: list[tuple[str, str]] = []

    # 1. Active session — orchestrator should drive that loop first.
    if active is not None:
        tool_map = {
            "autonomous": "autonomous_research_loop",
            "tree": "tree_autonomous_loop",
            "harness": "harness_autonomous_loop",
        }
        loop_tool = tool_map.get(active["kind"], "autonomous_research_loop")
        # If stage is a wait stage, hint status; otherwise hint continue/submit.
        wait_stages = {"benchmark_wait", "external_dispatch", "running"}
        is_wait = active.get("stage") in wait_stages
        primary = (
            loop_tool,
            (
                f"Active {active['kind']} session {active['session_id']} is in "
                f"stage {active['stage']!r}. "
                + (
                    f"Call {loop_tool}(action='status') to check progress, "
                    f"then 'continue' once external work (dispatch/collect) is done."
                    if is_wait
                    else
                    f"Call {loop_tool}(action='continue') to advance, or 'status' to inspect."
                )
            ),
        )
        alternatives.append((
            "get_fleet_status",
            "Inspect node-side progress while the session waits.",
        ))
        if active.get("kind") in {"autonomous", "tree"} and queue_sum["running"] > 0:
            alternatives.append((
                "collect_results",
                "Pull results so the session's pending nodes can finalize.",
            ))
        return _build_response(primary, alternatives, state)

    # 2. Orphan pods — clean up before any provision/bootstrap.
    if orphans_present:
        primary = _R_ORPHANS_PRESENT
        alternatives.append((
            "get_fleet_status",
            "List orphan details before destroying.",
        ))
        return _build_response(primary, alternatives, state)

    # 3 - 5. Fleet bootstrap chain.
    if nodes_sum["total"] == 0:
        primary = _R_NO_PODS
        alternatives.append((
            "list_projects",
            "If this isn't the project you expect, list configured projects first.",
        ))
        return _build_response(primary, alternatives, state)

    if nodes_sum["no_ssh"] > 0 and nodes_sum["ready"] == 0:
        primary = _R_PODS_NOT_REFRESHED
        alternatives.append((
            "get_fleet_status",
            "Inspect per-node state for stuck/failed pods.",
        ))
        return _build_response(primary, alternatives, state)

    if nodes_sum["unbootstrapped"] > 0 and nodes_sum["ready"] == 0:
        primary = _R_PODS_NOT_BOOTSTRAPPED
        alternatives.append((
            "get_fleet_status",
            "Verify nodes have SSH and inspect bootstrap state.",
        ))
        return _build_response(primary, alternatives, state)

    # 6. Experiments running — wait.
    if queue_sum["running"] > 0:
        primary = _R_RUNNING
        alternatives.append((
            "get_queue_status",
            "Check per-run progress.",
        ))
        alternatives.append((
            "collect_results",
            "Pull partial results as runs finish.",
        ))
        return _build_response(primary, alternatives, state)

    # 7. Queue has queued work + idle ready pods → dispatch.
    if queue_sum["queued"] > 0 and nodes_sum["ready"] > 0:
        primary = _R_DISPATCH
        alternatives.append((
            "get_queue_status",
            "Inspect what's queued before dispatching.",
        ))
        return _build_response(primary, alternatives, state)

    # 8. Queue empty + hypotheses pending → batch them.
    if queue_sum["queued"] == 0 and rs["pending_count"] > 0:
        primary = _R_QUEUE_EMPTY_HAVE_HYPS
        alternatives.append((
            "design_batch_from_hypotheses",
            "Build a batch from pending hypotheses first if you haven't.",
        ))
        return _build_response(primary, alternatives, state)

    # 9. Recent completions, no pending work, no active reflection step.
    # Recommend collecting if results haven't been pulled, else inspect
    # leaderboard.
    if queue_sum["finished"] > completed_count:
        # Some runs finished but local results store is behind.
        primary = _R_COLLECT
        alternatives.append((
            "get_leaderboard",
            "Inspect what's already been collected.",
        ))
        return _build_response(primary, alternatives, state)

    if completed_count > 0 and rs["pending_count"] == 0:
        # Have results, no pending hypotheses — time for leaderboard + reflect.
        primary = _R_LEADERBOARD_FRESH
        alternatives.append(_R_REFLECT)
        alternatives.append((
            "context_push_finding",
            "Record observations from the latest results as findings.",
        ))
        return _build_response(primary, alternatives, state)

    # 10. Nothing running, no completions, no hypotheses → kickstart.
    if completed_count == 0 and rs["pending_count"] == 0:
        primary = _R_NO_HYPOTHESES
        alternatives.append((
            "autonomous_research_loop",
            "Or start the full closed loop with action='start' if you want autonomy.",
        ))
        alternatives.append((
            "get_research_briefing",
            "Read the project state before deciding the first hypothesis.",
        ))
        return _build_response(primary, alternatives, state)

    # 11. Fallback — briefing.
    primary = _R_BRIEFING_DEFAULT
    return _build_response(primary, alternatives, state)


def _build_response(
    primary: tuple[str, str],
    alternatives: list[tuple[str, str]],
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "recommended_tool": primary[0],
        "rationale": primary[1],
        "alternatives": [
            {"tool": tool, "rationale": rationale}
            for tool, rationale in alternatives
        ],
        "state": state,
    }
