"""Tests for crucible.researcher.briefing.build_briefing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible import __version__ as CRUCIBLE_VERSION
from crucible.core.config import ProjectConfig
from crucible.researcher.briefing import build_briefing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_YAML = """\
name: briefing-test
version: "{version}"

metrics:
  primary: val_loss
  direction: minimize

researcher:
  budget_hours: 10.0
  max_iterations: 20

results_file: experiments.jsonl
fleet_results_file: experiments_fleet.jsonl
logs_dir: logs
store_dir: .crucible
research_state_file: research_state.jsonl
""".format(version=CRUCIBLE_VERSION)


def _make_result(
    name: str,
    val_loss: float,
    model_bytes: int = 50000,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": f"run_{name}",
        "name": name,
        "timestamp": "2025-06-01T00:00:00Z",
        "backend": "torch",
        "preset": "proxy",
        "config": {"LR": "0.001"},
        "result": {"val_loss": val_loss, "steps_completed": 1000},
        "model_bytes": model_bytes,
        "status": status,
        "tags": [],
        "error": None,
        "failure_class": None,
        "returncode": 0,
    }


def _setup_project(tmp_path: Path) -> ProjectConfig:
    """Create a minimal project directory and return its config."""
    yaml_path = tmp_path / "crucible.yaml"
    yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")

    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / ".crucible").mkdir(exist_ok=True)
    (tmp_path / "experiments.jsonl").touch()
    (tmp_path / "experiments_fleet.jsonl").touch()

    from crucible.core.config import load_config

    return load_config(yaml_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBriefingNoExperiments:
    """build_briefing should always succeed, even with no data."""

    def test_empty_project(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        briefing = build_briefing(config)

        # Structure checks -- all top-level keys present
        assert "project" in briefing
        assert "track" in briefing
        assert "recent_experiments" in briefing
        assert "leaderboard_top3" in briefing
        assert "active_hypotheses" in briefing
        assert "recent_findings" in briefing
        assert "recent_notes" in briefing
        assert "beliefs" in briefing
        assert "budget" in briefing
        assert "hub_findings" in briefing
        assert "suggested_next_steps" in briefing
        assert "markdown_summary" in briefing

    def test_project_section(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        briefing = build_briefing(config)

        assert briefing["project"]["name"] == "briefing-test"
        assert briefing["project"]["primary_metric"] == "val_loss"
        assert briefing["project"]["direction"] == "minimize"

    def test_empty_lists_when_no_data(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        briefing = build_briefing(config)

        assert briefing["recent_experiments"] == []
        assert briefing["leaderboard_top3"] == []
        assert briefing["active_hypotheses"] == []
        assert briefing["recent_findings"] == []
        assert briefing["recent_notes"] == []
        assert briefing["beliefs"] == []
        assert briefing["hub_findings"] == []

    def test_budget_defaults(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        briefing = build_briefing(config)

        assert briefing["budget"]["total_hours"] == 10.0
        assert briefing["budget"]["remaining_hours"] == 10.0

    def test_suggested_first_experiment(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        briefing = build_briefing(config)

        assert len(briefing["suggested_next_steps"]) > 0
        assert "first experiment" in briefing["suggested_next_steps"][0].lower()

    def test_markdown_summary_present(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        briefing = build_briefing(config)

        md = briefing["markdown_summary"]
        assert "Research Briefing" in md
        assert "briefing-test" in md
        assert "Workflow Guidance" in md


class TestBriefingWithExperiments:
    """build_briefing with experiment data."""

    def _populate_experiments(self, tmp_path: Path) -> None:
        results = [
            _make_result("exp-a", 1.50),
            _make_result("exp-b", 1.40),
            _make_result("exp-c", 1.35),
            _make_result("exp-d", 1.60, status="failed"),
        ]
        results_path = tmp_path / "experiments.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    def test_recent_experiments_populated(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_experiments(tmp_path)
        briefing = build_briefing(config)

        assert len(briefing["recent_experiments"]) > 0
        names = [e["name"] for e in briefing["recent_experiments"]]
        assert "exp-a" in names

    def test_leaderboard_top3(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_experiments(tmp_path)
        briefing = build_briefing(config)

        top3 = briefing["leaderboard_top3"]
        assert len(top3) <= 3
        if top3:
            # Should be sorted by val_loss ascending
            assert top3[0]["name"] == "exp-c"
            assert top3[0]["metric"] == 1.35

    def test_suggested_review_findings(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_experiments(tmp_path)
        briefing = build_briefing(config)

        # We have experiments but no findings, so should suggest reviewing
        steps = briefing["suggested_next_steps"]
        assert any("finding" in s.lower() or "review" in s.lower() for s in steps)


class TestBriefingWithFindings:
    """build_briefing with research state findings."""

    def _populate_state(self, tmp_path: Path) -> None:
        from crucible.researcher.state import ResearchState

        state_path = tmp_path / "research_state.jsonl"
        state = ResearchState(state_path, budget_hours=10.0)
        state.add_finding(
            "Lower learning rates are more stable",
            category="observation",
            confidence=0.8,
        )
        state.add_finding(
            "Batch size 4096 tokens is optimal",
            category="belief",
            confidence=0.7,
        )
        state.add_hypothesis({
            "name": "test-lr-warmup",
            "hypothesis": "LR warmup reduces early instability",
            "expected_impact": 0.05,
            "status": "pending",
        })
        state.update_beliefs(["Lower LR is better", "Warmup helps"])
        state.save()

    def test_findings_present(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_state(tmp_path)
        briefing = build_briefing(config)

        assert len(briefing["recent_findings"]) == 2
        assert briefing["recent_findings"][0]["category"] == "observation"

    def test_hypotheses_present(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_state(tmp_path)
        briefing = build_briefing(config)

        assert len(briefing["active_hypotheses"]) == 1
        assert briefing["active_hypotheses"][0]["name"] == "test-lr-warmup"

    def test_beliefs_present(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_state(tmp_path)
        briefing = build_briefing(config)

        assert len(briefing["beliefs"]) == 2
        assert "Lower LR is better" in briefing["beliefs"]

    def test_pending_hypothesis_in_suggestions(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_state(tmp_path)

        # Also add experiments so the "first experiment" suggestion doesn't dominate
        results = [_make_result("exp-a", 1.50)]
        results_path = tmp_path / "experiments.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        briefing = build_briefing(config)
        steps = briefing["suggested_next_steps"]
        assert any("test-lr-warmup" in s for s in steps)


class TestBriefingWithNotes:
    """build_briefing with experiment notes."""

    def _populate_notes(self, tmp_path: Path) -> None:
        from crucible.runner.notes import NoteStore

        store = NoteStore(tmp_path / ".crucible")
        store.add(
            run_id="run_exp-a",
            body="This run looked promising, LR seemed stable.",
            stage="post-run",
            tags=["analysis"],
        )
        store.add(
            run_id="run_exp-b",
            body="Trying lower LR to see if loss improves.",
            stage="pre-run",
            tags=["hypothesis"],
        )

    def test_notes_present(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_notes(tmp_path)
        briefing = build_briefing(config)

        assert len(briefing["recent_notes"]) == 2


class TestBriefingWithHub:
    """build_briefing with hub findings (uses temp hub dir)."""

    def _populate_hub(self, tmp_path: Path) -> "tuple[Any, str]":
        from crucible.core.hub import HubStore

        hub_dir = tmp_path / "test-hub"
        hub = HubStore.init(hub_dir, name="test-hub")
        hub.create_track("ml-opt", description="ML optimization research")
        hub.activate_track("ml-opt")

        hub.store_finding(
            {"title": "AdamW outperforms SGD", "body": "Consistent across experiments."},
            scope="track",
            track="ml-opt",
        )

        return hub, str(hub_dir)

    def test_hub_findings_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _setup_project(tmp_path)
        _, hub_dir = self._populate_hub(tmp_path)
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", hub_dir)

        briefing = build_briefing(config)
        assert len(briefing["hub_findings"]) >= 1

    def test_track_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _setup_project(tmp_path)
        _, hub_dir = self._populate_hub(tmp_path)
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", hub_dir)

        briefing = build_briefing(config)
        assert briefing["track"] is not None
        assert briefing["track"]["name"] == "ml-opt"


class TestBriefingBudgetLow:
    """Suggested actions when budget is low."""

    def _populate_state_low_budget(self, tmp_path: Path) -> None:
        from crucible.researcher.state import ResearchState

        state_path = tmp_path / "research_state.jsonl"
        state = ResearchState(state_path, budget_hours=10.0)
        # Use up 9 hours
        state.charge_hours(9.0)
        state.save()

    def test_low_budget_suggestion(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        self._populate_state_low_budget(tmp_path)

        # Add experiments so we don't get the "first experiment" suggestion
        results = [_make_result("exp-a", 1.50)]
        with open(tmp_path / "experiments.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        briefing = build_briefing(config)
        steps = briefing["suggested_next_steps"]
        assert any("budget" in s.lower() or "promotion" in s.lower() for s in steps)


class TestBriefingNeverFails:
    """build_briefing should never raise, even with corrupt data."""

    def test_corrupt_experiments_jsonl(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        # Write garbage to experiments file
        (tmp_path / "experiments.jsonl").write_text("not json\nstill not json\n")

        # Should not raise
        briefing = build_briefing(config)
        assert "project" in briefing
        assert "markdown_summary" in briefing

    def test_corrupt_research_state(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        (tmp_path / "research_state.jsonl").write_text("{{{{bad json\n")

        # Should not raise (state loader may skip bad lines)
        briefing = build_briefing(config)
        assert "project" in briefing

    def test_missing_store_dir(self, tmp_path: Path) -> None:
        config = _setup_project(tmp_path)
        # Remove .crucible dir
        import shutil
        shutil.rmtree(tmp_path / ".crucible", ignore_errors=True)

        briefing = build_briefing(config)
        assert "project" in briefing
        assert briefing["recent_notes"] == []
