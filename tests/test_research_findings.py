"""Tests for ResearchState findings support."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.researcher.state import ResearchState


class TestFindings:
    def test_add_finding(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        entry = state.add_finding("Width matters more than depth", category="belief")
        assert entry["finding"] == "Width matters more than depth"
        assert entry["category"] == "belief"
        assert len(state.findings) == 1

    def test_get_findings_all(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_finding("Finding A", category="observation")
        state.add_finding("Finding B", category="belief")
        findings = state.get_findings()
        assert len(findings) == 2

    def test_get_findings_filtered(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_finding("Finding A", category="observation")
        state.add_finding("Finding B", category="belief")
        state.add_finding("Finding C", category="observation")
        filtered = state.get_findings(category="observation")
        assert len(filtered) == 2
        assert all(f["category"] == "observation" for f in filtered)

    def test_get_findings_with_limit(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        for i in range(10):
            state.add_finding(f"Finding {i}")
        limited = state.get_findings(limit=3)
        assert len(limited) == 3

    def test_findings_persist(self, tmp_path):
        path = tmp_path / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.add_finding("Persistent finding", category="constraint", confidence=0.9)
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        assert len(reloaded.findings) == 1
        assert reloaded.findings[0]["finding"] == "Persistent finding"
        assert reloaded.findings[0]["category"] == "constraint"
        assert reloaded.findings[0]["confidence"] == 0.9

    def test_finding_defaults(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        entry = state.add_finding("Simple finding")
        assert entry["category"] == "observation"
        assert entry["confidence"] == 0.7
        assert entry["source_experiments"] == []
        assert entry["created_by"] == "unknown"
        assert "ts" in entry
