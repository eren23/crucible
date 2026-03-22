"""FastAPI route definitions for the Crucible REST API.

All routes delegate to the same backends as MCP tools (NoteStore, VersionStore,
analysis modules) to ensure data consistency across entry points.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

from crucible.api.models import (
    FindingCreate,
    HealthResponse,
    NoteCreate,
)
from crucible.core.config import ProjectConfig, load_config
from crucible.runner.notes import NoteStore

router = APIRouter()

_start_time = time.time()


def _get_config() -> ProjectConfig:
    return load_config()


def _get_note_store() -> NoteStore:
    config = _get_config()
    return NoteStore(config.project_root / config.store_dir)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict[str, Any]:
    config = _get_config()
    return {
        "status": "ok",
        "version": config.version,
        "project": config.name,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@router.get("/experiments")
def list_experiments(
    name: str = "",
    family: str = "",
    tag: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    config = _get_config()
    try:
        from crucible.analysis.results import completed_results, merged_results

        results = merged_results(config)
        if name:
            results = [r for r in results if name.lower() in r.get("name", "").lower()]
        if family:
            results = [r for r in results if r.get("config", {}).get("MODEL_FAMILY") == family]
        if tag:
            results = [r for r in results if tag in r.get("tags", [])]
        entries = []
        for r in results[:limit]:
            res = r.get("result", {}) or {}
            entries.append({
                "name": r.get("name", ""),
                "run_id": r.get("id", ""),
                "status": r.get("status", ""),
                "primary_value": res.get(config.metrics.primary),
                "model_bytes": r.get("model_bytes"),
                "tags": r.get("tags", []),
                "timestamp": r.get("timestamp", ""),
            })
        return {"experiments": entries, "total": len(entries)}
    except Exception as exc:
        return {"experiments": [], "total": 0, "note": str(exc)}


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str) -> dict[str, Any]:
    config = _get_config()
    try:
        from crucible.analysis.results import merged_results

        results = merged_results(config)
        for r in results:
            if r.get("id") == run_id or r.get("name") == run_id:
                return {"found": True, "result": r}
        return {"found": False, "result": {}}
    except Exception as exc:
        return {"found": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


@router.get("/experiments/{run_id}/notes")
def get_experiment_notes(run_id: str) -> dict[str, Any]:
    note_store = _get_note_store()
    notes = note_store.get_for_run(run_id)
    return {"run_id": run_id, "notes": notes, "count": len(notes)}


@router.post("/experiments/{run_id}/notes")
def add_experiment_note(run_id: str, data: NoteCreate) -> dict[str, Any]:
    if not data.text:
        return {"error": "Note body is required."}
    note_store = _get_note_store()
    meta = note_store.add(
        run_id,
        body=data.text,
        tags=data.tags,
        stage=data.stage,
        created_by=data.created_by,
    )
    return {"status": "created", "note": meta}


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@router.get("/findings")
def list_findings(category: str = "", limit: int = 50) -> dict[str, Any]:
    config = _get_config()
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)
        findings = state.get_findings(category=category or None, limit=limit)
        return {"findings": findings, "total": len(findings)}
    except Exception as exc:
        return {"findings": [], "error": str(exc)}


@router.post("/findings")
def push_finding(data: FindingCreate) -> dict[str, Any]:
    config = _get_config()
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)
        state.add_finding(
            finding=data.finding,
            category=data.category,
            source_experiments=data.source_experiments,
            confidence=data.confidence,
            created_by=data.created_by,
        )
        state.save()
        return {"status": "recorded"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


@router.get("/tracks")
def list_tracks() -> dict[str, Any]:
    try:
        from crucible.core.hub import HubStore

        hub = HubStore()
        if not hub.initialized:
            return {"tracks": [], "note": "Hub not initialized"}
        tracks = hub.list_tracks()
        active = hub.get_active_track()
        return {"tracks": tracks, "active_track": active, "total": len(tracks)}
    except Exception as exc:
        return {"tracks": [], "error": str(exc)}


@router.get("/tracks/{name}/briefing")
def track_briefing(name: str) -> dict[str, Any]:
    config = _get_config()
    try:
        from crucible.researcher.briefing import build_briefing

        config.active_track = name
        return build_briefing(config)
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


@router.get("/leaderboard")
def leaderboard(top_n: int = 20) -> dict[str, Any]:
    config = _get_config()
    try:
        from crucible.analysis.leaderboard import leaderboard as lb
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        top = lb(results, top_n=top_n, cfg=config)
        entries = []
        for i, r in enumerate(top, 1):
            res = r.get("result", {}) or {}
            entries.append({
                "rank": i,
                "name": r.get("name", ""),
                "primary_value": res.get(config.metrics.primary),
                "model_bytes": r.get("model_bytes"),
            })
        return {"total_completed": len(results), "top": entries}
    except Exception as exc:
        return {"top": [], "error": str(exc)}
