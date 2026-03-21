"""Tests for crucible.researcher.state."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.researcher.state import ResearchState


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestEmptyState:
    def test_budget_remaining(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.budget_remaining == 5.0

    def test_empty_hypotheses(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.hypotheses == []

    def test_empty_history(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.history == []

    def test_empty_beliefs(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.beliefs == []


# ---------------------------------------------------------------------------
# Hypothesis management
# ---------------------------------------------------------------------------

class TestHypothesisManagement:
    def test_add_hypothesis(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "test", "expected_impact": 0.5, "config": {"A": "1"}})
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0]["name"] == "test"
        assert state.hypotheses[0]["status"] == "pending"

    def test_hypotheses_sorted_by_impact(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "low", "expected_impact": 0.1, "config": {}})
        state.add_hypothesis({"name": "high", "expected_impact": 0.9, "config": {}})
        state.add_hypothesis({"name": "mid", "expected_impact": 0.5, "config": {}})
        # Highest impact first
        assert state.hypotheses[0]["name"] == "high"
        assert state.hypotheses[1]["name"] == "mid"
        assert state.hypotheses[2]["name"] == "low"

    def test_pending_hypotheses(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "a", "expected_impact": 0.5, "config": {}})
        state.add_hypothesis({"name": "b", "expected_impact": 0.3, "config": {}})
        state.mark_hypothesis("a", "tested")
        pending = state.pending_hypotheses()
        assert len(pending) == 1
        assert pending[0]["name"] == "b"

    def test_mark_hypothesis(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"hypothesis": "h1", "expected_impact": 0.5, "config": {}})
        state.mark_hypothesis("h1", "tested")
        assert state.hypotheses[0]["status"] == "tested"

    def test_add_hypothesis_sets_timestamp(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "ts_test", "config": {}})
        assert "ts" in state.hypotheses[0]

    def test_hypotheses_sorted_by_expected_bpb_impact(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "a", "expected_bpb_impact": 0.2, "config": {}})
        state.add_hypothesis({"name": "b", "expected_bpb_impact": 0.8, "config": {}})
        assert state.hypotheses[0]["name"] == "b"


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

class TestBudgetTracking:
    def test_record_result_charges_budget(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 2.0},
            result={"status": "completed", "val_bpb": 1.2},
        )
        assert state.budget_remaining == 8.0

    def test_multiple_charges(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 3.0},
            result={"status": "completed"},
        )
        state.record_result(
            experiment={"name": "exp2", "pod_hours": 4.0},
            result={"status": "completed"},
        )
        assert state.budget_remaining == 3.0

    def test_budget_cannot_go_negative(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=1.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 5.0},
            result={"status": "completed"},
        )
        assert state.budget_remaining == 0.0

    def test_charge_hours(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.charge_hours(2.5)
        assert state.budget_remaining == 7.5


# ---------------------------------------------------------------------------
# Beliefs
# ---------------------------------------------------------------------------

class TestBeliefs:
    def test_update_beliefs(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.update_beliefs(["belief A", "belief B"])
        assert state.beliefs == ["belief A", "belief B"]

    def test_update_beliefs_replaces(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.update_beliefs(["old"])
        state.update_beliefs(["new1", "new2"])
        assert state.beliefs == ["new1", "new2"]


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_record_result_appends_history(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 1.0},
            result={"status": "completed", "val_bpb": 1.5},
        )
        assert len(state.history) == 1
        assert state.history[0]["experiment"]["name"] == "exp1"
        assert state.history[0]["result"]["val_bpb"] == 1.5

    def test_get_history_summary_empty(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        summary = state.get_history_summary()
        assert "No experiments completed yet" in summary

    def test_get_history_summary_with_data(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 1.0},
            result={"status": "completed", "val_loss": 1.2},
        )
        summary = state.get_history_summary()
        assert "exp1" in summary
        assert "1.2" in summary


# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------

class TestSaveAndReload:
    def test_full_save_reload_cycle(self, tmp_path):
        path = tmp_path / "state.jsonl"
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

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.save()
        assert path.exists()

    def test_reload_preserves_hypothesis_order(self, tmp_path):
        path = tmp_path / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.add_hypothesis({"name": "low", "expected_impact": 0.1, "config": {}})
        state.add_hypothesis({"name": "high", "expected_impact": 0.9, "config": {}})
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        # After reload, hypotheses are loaded in file order (not re-sorted)
        # The file stores them in the sorted order from add_hypothesis
        assert reloaded.hypotheses[0]["name"] == "high"

    def test_reload_budget_adjustment(self, tmp_path):
        path = tmp_path / "state.jsonl"
        state = ResearchState(path, budget_hours=20.0)
        state.charge_hours(5.0)
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        # Budget adjustment from file overrides constructor param
        assert reloaded.budget_remaining == 15.0
