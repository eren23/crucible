"""Research briefing: composable session-orientation snapshot.

Aggregates data from experiments, leaderboard, research state, notes,
and hub findings into a single dict for LLM consumption.  Every section
is wrapped in try/except so the function never fails -- missing data is
simply omitted or given safe defaults.
"""
from __future__ import annotations

from typing import Any

from crucible.core.config import ProjectConfig


# ---------------------------------------------------------------------------
# Section builders (each returns a safe dict slice)
# ---------------------------------------------------------------------------


def _project_section(config: ProjectConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "primary_metric": config.metrics.primary,
        "direction": config.metrics.direction,
    }


def _track_section(config: ProjectConfig) -> dict[str, Any] | None:
    """Active research track info from hub, if available."""
    try:
        from crucible.core.hub import HubStore

        hub = HubStore()
        if not hub.initialized:
            return None
        active = hub.get_active_track()
        if not active:
            # Fall back to config-level active_track
            active = config.active_track or None
            if not active:
                return None
        track = hub.get_track(active)
        if track is None:
            return None
        return {"name": track.get("name", active), "description": track.get("description", "")}
    except Exception:
        if config.active_track:
            return {"name": config.active_track, "description": ""}
        return None


def _recent_experiments(config: ProjectConfig, limit: int = 10) -> list[dict[str, Any]]:
    try:
        from crucible.analysis.results import merged_results

        all_results = merged_results(config)
        # Sort by timestamp descending
        all_results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        recent = all_results[:limit]
        entries = []
        primary = config.metrics.primary
        for r in recent:
            res = r.get("result") or {}
            entries.append({
                "name": r.get("name", ""),
                "run_id": r.get("id", ""),
                "status": r.get("status", "unknown"),
                "metric": res.get(primary),
                "timestamp": r.get("timestamp", ""),
            })
        return entries
    except Exception:
        return []


def _leaderboard_top3(config: ProjectConfig) -> list[dict[str, Any]]:
    try:
        from crucible.analysis.leaderboard import leaderboard
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        top = leaderboard(results, top_n=3, cfg=config)
        primary = config.metrics.primary
        entries = []
        for r in top:
            res = r.get("result") or {}
            entries.append({
                "name": r.get("name", ""),
                "metric": res.get(primary),
                "model_bytes": r.get("model_bytes"),
            })
        return entries
    except Exception:
        return []


def _hypotheses_section(config: ProjectConfig) -> list[dict[str, Any]]:
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        if not state_path.exists():
            return []
        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)
        entries = []
        for h in state.hypotheses:
            entries.append({
                "name": h.get("name", h.get("hypothesis", "")),
                "expected_impact": h.get("expected_impact", h.get("expected_bpb_impact", 0.0)),
                "status": h.get("status", "pending"),
            })
        return entries
    except Exception:
        return []


def _recent_findings(config: ProjectConfig, limit: int = 10) -> list[dict[str, Any]]:
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        if not state_path.exists():
            return []
        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)
        findings = state.get_findings(limit=limit)
        entries = []
        for f in findings:
            entries.append({
                "finding": f.get("finding", ""),
                "category": f.get("category", "observation"),
                "confidence": f.get("confidence", 0.5),
                "ts": f.get("ts", ""),
            })
        return entries
    except Exception:
        return []


def _recent_notes(config: ProjectConfig, limit: int = 10) -> list[dict[str, Any]]:
    try:
        from crucible.runner.notes import NoteStore

        store_dir = config.project_root / config.store_dir
        store = NoteStore(store_dir)
        all_notes = store.search(limit=limit)
        entries = []
        for n in all_notes:
            # Load actual body content for meaningful preview
            body_preview = ""
            note_id = n.get("note_id", "")
            if note_id:
                result = store.get_note(note_id)
                if result is not None:
                    _, body = result
                    body_preview = body[:120]
            entries.append({
                "run_id": n.get("run_id", ""),
                "note_id": note_id,
                "stage": n.get("stage", ""),
                "body_preview": body_preview,
            })
        return entries
    except Exception:
        return []


def _beliefs_section(config: ProjectConfig) -> list[str]:
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        if not state_path.exists():
            return []
        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)
        return list(state.beliefs)
    except Exception:
        return []


def _budget_section(config: ProjectConfig) -> dict[str, float]:
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        if not state_path.exists():
            return {
                "remaining_hours": config.researcher.budget_hours,
                "total_hours": config.researcher.budget_hours,
            }
        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)
        return {
            "remaining_hours": state.budget_remaining,
            "total_hours": config.researcher.budget_hours,
        }
    except Exception:
        return {
            "remaining_hours": config.researcher.budget_hours,
            "total_hours": config.researcher.budget_hours,
        }


def _hub_findings(config: ProjectConfig) -> list[dict[str, Any]]:
    """Load track + global findings from hub, if available."""
    try:
        from crucible.core.hub import HubStore

        hub = HubStore()
        if not hub.initialized:
            return []

        active_track = hub.get_active_track() or config.active_track
        if active_track:
            return hub.load_context_for_track(active_track, include_global=True, max_findings=20)
        else:
            # No active track -- just load global findings
            return hub.list_findings("global", status="active")
    except Exception:
        return []


def _suggested_next_steps(
    experiments: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    budget: dict[str, float],
) -> list[str]:
    """Generate suggested next steps based on current state."""
    steps: list[str] = []

    if not experiments:
        steps.append("Run your first experiment with `enqueue_experiment` or `version_run_design`.")
        return steps

    if not findings:
        steps.append("Review recent results and record findings with `context_push_finding`.")

    pending = [h for h in hypotheses if h.get("status") == "pending"]
    if pending:
        top = pending[0]
        name = top.get("name", "unnamed")
        steps.append(f"Test pending hypothesis: {name}")

    remaining = budget.get("remaining_hours", 0.0)
    total = budget.get("total_hours", 1.0)
    if total > 0 and remaining / total < 0.2:
        steps.append(
            "Budget running low -- focus on promotion-tier validation of best config."
        )

    if not steps:
        steps.append("Continue iterating: review leaderboard, form hypotheses, run experiments.")

    return steps


def _markdown_summary(
    project: dict[str, Any],
    track: dict[str, Any] | None,
    experiments: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    beliefs: list[str],
    budget: dict[str, float],
    suggested: list[str],
) -> str:
    """Render a human-readable markdown summary."""
    lines: list[str] = []
    lines.append(f"# Research Briefing: {project.get('name', 'Unknown')}")
    lines.append("")

    if track:
        lines.append(f"**Active Track:** {track.get('name', 'none')}")
        if track.get("description"):
            lines.append(f"  {track['description']}")
        lines.append("")

    # Budget
    rem = budget.get("remaining_hours", 0.0)
    tot = budget.get("total_hours", 0.0)
    lines.append(f"**Budget:** {rem:.1f}h remaining of {tot:.1f}h total")
    lines.append("")

    # Leaderboard
    primary = project.get("primary_metric", "val_loss")
    if leaderboard:
        lines.append("## Leaderboard (Top 3)")
        for i, entry in enumerate(leaderboard, 1):
            metric_val = entry.get("metric")
            metric_str = f"{metric_val:.4f}" if isinstance(metric_val, (int, float)) else str(metric_val)
            lines.append(f"{i}. **{entry.get('name', '?')}** {primary}={metric_str} ({entry.get('model_bytes', '?')} bytes)")
        lines.append("")
    else:
        lines.append("## Leaderboard")
        lines.append("No completed experiments yet.")
        lines.append("")

    # Recent experiments
    if experiments:
        lines.append(f"## Recent Experiments ({len(experiments)})")
        for exp in experiments[:5]:
            metric_val = exp.get("metric")
            metric_str = f"{metric_val:.4f}" if isinstance(metric_val, (int, float)) else str(metric_val)
            lines.append(f"- {exp.get('name', '?')} [{exp.get('status', '?')}] {primary}={metric_str}")
        lines.append("")

    # Beliefs
    if beliefs:
        lines.append("## Current Beliefs")
        for belief in beliefs:
            lines.append(f"- {belief}")
        lines.append("")

    # Hypotheses
    pending = [h for h in hypotheses if h.get("status") == "pending"]
    if pending:
        lines.append(f"## Pending Hypotheses ({len(pending)})")
        for h in pending[:5]:
            lines.append(f"- {h.get('name', '?')} (expected impact: {h.get('expected_impact', '?')})")
        lines.append("")

    # Findings
    if findings:
        lines.append(f"## Recent Findings ({len(findings)})")
        for f in findings[:5]:
            lines.append(f"- [{f.get('category', '?')}] {f.get('finding', '?')[:100]}")
        lines.append("")

    # Notes
    if notes:
        lines.append(f"## Recent Notes ({len(notes)})")
        for n in notes[:5]:
            lines.append(f"- [{n.get('stage', '?')}] run={n.get('run_id', '?')} {n.get('note_id', '')}")
        lines.append("")

    # Suggested actions
    lines.append("## Suggested Actions")
    for step in suggested:
        lines.append(f"- {step}")
    lines.append("")

    # Workflow guidance
    lines.append("## Workflow Guidance")
    lines.append("- **Pre-run:** use `note_add` with `stage=\"pre-run\"` to record your hypothesis")
    lines.append("- **During-run:** use `note_add` with `stage=\"during-run\"` for observations")
    lines.append("- **Post-run:** use `note_add` with `stage=\"post-run\"` + `context_push_finding` for synthesis")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_briefing(config: ProjectConfig) -> dict[str, Any]:
    """Compose a research briefing from existing sources -- no new data path.

    Every section is fetched independently with try/except so the briefing
    always succeeds even if individual data sources are unavailable.
    """
    project = _project_section(config)
    track = _track_section(config)
    experiments = _recent_experiments(config)
    top3 = _leaderboard_top3(config)
    hypotheses = _hypotheses_section(config)
    findings = _recent_findings(config)
    notes = _recent_notes(config)
    beliefs = _beliefs_section(config)
    budget = _budget_section(config)
    hub = _hub_findings(config)
    suggested = _suggested_next_steps(experiments, findings, hypotheses, budget)
    summary = _markdown_summary(
        project, track, experiments, top3, hypotheses,
        findings, notes, beliefs, budget, suggested,
    )

    return {
        "project": project,
        "track": track,
        "recent_experiments": experiments,
        "leaderboard_top3": top3,
        "active_hypotheses": hypotheses,
        "recent_findings": findings,
        "recent_notes": notes,
        "beliefs": beliefs,
        "budget": budget,
        "hub_findings": hub,
        "suggested_next_steps": suggested,
        "markdown_summary": summary,
    }
