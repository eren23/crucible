"""Pydantic request models for the Crucible API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class NoteCreate(BaseModel):
    """Request body for creating a note."""

    text: str
    stage: str = ""
    tags: list[str] = Field(default_factory=list)
    created_by: str = "api"


class FindingCreate(BaseModel):
    """Request body for recording a finding."""

    finding: str
    category: str = "observation"
    source_experiments: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    created_by: str = "api"
