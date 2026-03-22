"""Pydantic response/request models for the Crucible API."""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class ExperimentSummary(BaseModel):
    """Summary view of a single experiment result."""

    name: str = ""
    run_id: str = ""
    status: str = ""
    primary_metric: str = ""
    primary_value: Optional[float] = None
    model_bytes: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    timestamp: str = ""


class ExperimentDetail(BaseModel):
    """Full experiment result with all fields."""

    found: bool = True
    result: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class NoteEntry(BaseModel):
    """A single experiment note."""

    run_id: str = ""
    text: str = ""
    stage: str = ""
    tags: List[str] = Field(default_factory=list)
    created_at: str = ""
    created_by: str = ""


class NoteCreate(BaseModel):
    """Request body for creating a note."""

    text: str
    stage: str = ""
    tags: List[str] = Field(default_factory=list)
    created_by: str = "api"


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class FindingEntry(BaseModel):
    """A single research finding."""

    finding: str = ""
    category: str = "observation"
    source_experiments: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    created_by: str = ""
    ts: str = ""


class FindingCreate(BaseModel):
    """Request body for recording a finding."""

    finding: str
    category: str = "observation"
    source_experiments: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    created_by: str = "api"


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class TrackSummary(BaseModel):
    """Summary of a research track."""

    name: str = ""
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    status: str = "active"
    created_at: str = ""
    experiment_count: int = 0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    project: str = ""
