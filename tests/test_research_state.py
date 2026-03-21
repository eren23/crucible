"""Tests for crucible.researcher.state."""
import tempfile
from pathlib import Path

from crucible.researcher.state import ResearchState


def test_empty_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.jsonl"
        state = ResearchState(path, budget_hours=5.0)
        assert state.budget_remaining == 5.0
        assert state.hypotheses == []
        assert state.history == []
        assert state.beliefs == []


def test_add_hypothesis():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.add_hypothesis({"name": "test", "expected_impact": 0.5, "config": {"A": "1"}})
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0]["name"] == "test"
        assert state.hypotheses[0]["status"] == "pending"


def test_record_result_charges_budget():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 2.0},
            result={"status": "completed", "val_bpb": 1.2},
        )
        assert state.budget_remaining == 8.0
        assert len(state.history) == 1


def test_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.add_hypothesis({"name": "h1", "expected_impact": 0.3, "config": {"X": "1"}})
        state.update_beliefs(["belief 1", "belief 2"])
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 1.0},
            result={"status": "completed"},
        )
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        assert len(reloaded.hypotheses) == 1
        assert len(reloaded.history) == 1
        assert reloaded.beliefs == ["belief 1", "belief 2"]
        assert reloaded.budget_remaining == 9.0
