"""Tests for the autonomous_research_loop session driver (Phase 1.1)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError, StaleSubmitError
from crucible.researcher import autonomous_session as autos
from crucible.researcher.autonomous_session import (
    AutonomousSession,
    AutonomousSessionError,
    DoomLoopDetected,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_config(project_dir: Path):
    """ProjectConfig loaded from the conftest project_dir fixture."""
    os.chdir(project_dir)  # load_config reads cwd
    (project_dir / "program.md").write_text("Minimise val_loss.", encoding="utf-8")
    return load_config()


def _canned_hypothesis_response() -> dict[str, Any]:
    return {
        "hypotheses": [
            {
                "hypothesis": "test hypothesis",
                "name": "h_test",
                "expected_impact": 0.01,
                "confidence": 0.5,
                "config": {"MODEL_FAMILY": "baseline"},
                "rationale": "test",
                "family": "baseline",
            }
        ]
    }


def _canned_reflection_response() -> dict[str, Any]:
    return {
        "beliefs": ["b1", "b2"],
        "surprises": [],
        "promote": [],
        "kill": [],
    }


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


class TestStart:
    def test_creates_session_and_returns_first_prompt(self, project_config):
        out = autos.action_start(project_config, iterations=3, tier="proxy")
        assert "session_id" in out
        assert out["stage"] == AutonomousSession.STAGE_HYPOTHESIS
        assert out["iteration"] == 0
        assert isinstance(out["system"], str) and len(out["system"]) > 0
        assert isinstance(out["user"], str)
        assert "state_snapshot" in out
        assert out["session_status"] == AutonomousSession.STATUS_RUNNING
        assert out["iterations_planned"] == 3

    def test_rejects_zero_iterations(self, project_config):
        with pytest.raises(CrucibleError, match="iterations must be >= 1"):
            autos.action_start(project_config, iterations=0, tier="proxy")

    def test_persists_session_file(self, project_config):
        out = autos.action_start(project_config, iterations=2, tier="proxy")
        sid = out["session_id"]
        path = project_config.project_root / ".crucible" / "autonomous_sessions" / f"{sid}.yaml"
        assert path.exists()

    def test_idempotent_second_start_returns_active_session(self, project_config):
        first = autos.action_start(project_config, iterations=3, tier="proxy")
        second = autos.action_start(project_config, iterations=3, tier="proxy")
        assert first["session_id"] == second["session_id"]


# ---------------------------------------------------------------------------
# submit — happy path
# ---------------------------------------------------------------------------


class TestSubmitFlow:
    def test_hypothesis_advances_to_reflection(self, project_config):
        first = autos.action_start(project_config, iterations=2, tier="proxy")
        sid = first["session_id"]
        result = autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_hypothesis_response(),
            state_snapshot=first["state_snapshot"],
        )
        assert result["stage_applied"] == AutonomousSession.STAGE_HYPOTHESIS
        assert result["next_stage"] == AutonomousSession.STAGE_REFLECTION
        assert result["session_status"] == AutonomousSession.STATUS_RUNNING
        assert result["next_prompt"]["stage"] == AutonomousSession.STAGE_REFLECTION

    def test_reflection_advances_to_next_iteration_hypothesis(self, project_config):
        first = autos.action_start(project_config, iterations=2, tier="proxy")
        sid = first["session_id"]
        # Hypothesis stage
        afterh = autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_hypothesis_response(),
            state_snapshot=first["state_snapshot"],
        )
        # Reflection stage
        reflection_snap = afterh["next_prompt"]["state_snapshot"]
        afterr = autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_reflection_response(),
            state_snapshot=reflection_snap,
        )
        assert afterr["stage_applied"] == AutonomousSession.STAGE_REFLECTION
        assert afterr["next_stage"] == AutonomousSession.STAGE_HYPOTHESIS
        assert afterr["iterations_completed"] == 1
        assert afterr["next_prompt"]["iteration"] == 1
        assert afterr["next_prompt"]["stage"] == AutonomousSession.STAGE_HYPOTHESIS

    def test_session_done_after_planned_iterations(self, project_config):
        first = autos.action_start(project_config, iterations=1, tier="proxy")
        sid = first["session_id"]
        afterh = autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_hypothesis_response(),
            state_snapshot=first["state_snapshot"],
        )
        reflection_snap = afterh["next_prompt"]["state_snapshot"]
        afterr = autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_reflection_response(),
            state_snapshot=reflection_snap,
        )
        assert afterr["session_status"] == AutonomousSession.STATUS_DONE
        assert afterr["next_prompt"] is None
        assert afterr["iterations_completed"] == 1


# ---------------------------------------------------------------------------
# submit — error paths
# ---------------------------------------------------------------------------


class TestSubmitErrors:
    def test_stale_snapshot_raises(self, project_config):
        first = autos.action_start(project_config, iterations=2, tier="proxy")
        sid = first["session_id"]
        stale_snapshot = first["state_snapshot"]

        # Simulate concurrent state mutation persisted to disk.
        from crucible.researcher.state import ResearchState
        state = ResearchState(
            project_config.project_root / project_config.research_state_file,
            budget_hours=project_config.researcher.budget_hours,
        )
        state.add_finding("interloper", confidence=0.7)
        state.save()

        with pytest.raises(StaleSubmitError):
            autos.action_submit(
                project_config,
                session_id=sid,
                response=_canned_hypothesis_response(),
                state_snapshot=stale_snapshot,
            )

    def test_cannot_submit_to_done_session(self, project_config):
        first = autos.action_start(project_config, iterations=1, tier="proxy")
        sid = first["session_id"]
        afterh = autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_hypothesis_response(),
            state_snapshot=first["state_snapshot"],
        )
        autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_reflection_response(),
            state_snapshot=afterh["next_prompt"]["state_snapshot"],
        )
        # session is now done
        with pytest.raises(AutonomousSessionError, match="done"):
            autos.action_submit(
                project_config,
                session_id=sid,
                response=_canned_hypothesis_response(),
            )

    def test_unknown_session_raises(self, project_config):
        with pytest.raises(AutonomousSessionError, match="not found"):
            autos.action_status(project_config, session_id="not-a-real-uuid")


# ---------------------------------------------------------------------------
# status + cancel
# ---------------------------------------------------------------------------


class TestStatusAndCancel:
    def test_status_returns_state(self, project_config):
        first = autos.action_start(project_config, iterations=3, tier="proxy")
        sid = first["session_id"]
        status = autos.action_status(project_config, session_id=sid)
        assert status["session_id"] == sid
        assert status["status"] == AutonomousSession.STATUS_RUNNING
        assert status["iterations_planned"] == 3
        assert "yaml_path" in status
        assert "jsonl_path" in status

    def test_cancel_marks_session_canceled(self, project_config):
        first = autos.action_start(project_config, iterations=3, tier="proxy")
        sid = first["session_id"]
        out = autos.action_cancel(project_config, session_id=sid, reason="testing")
        assert out["session_status"] == AutonomousSession.STATUS_CANCELED
        assert out["already_terminal"] is False
        # Idempotent: second cancel reports already_terminal
        again = autos.action_cancel(project_config, session_id=sid)
        assert again["already_terminal"] is True

    def test_cannot_submit_after_cancel(self, project_config):
        first = autos.action_start(project_config, iterations=2, tier="proxy")
        sid = first["session_id"]
        autos.action_cancel(project_config, session_id=sid)
        with pytest.raises(AutonomousSessionError, match="canceled"):
            autos.action_submit(
                project_config,
                session_id=sid,
                response=_canned_hypothesis_response(),
            )


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_events_appended_to_jsonl(self, project_config):
        first = autos.action_start(project_config, iterations=1, tier="proxy")
        sid = first["session_id"]
        autos.action_submit(
            project_config,
            session_id=sid,
            response=_canned_hypothesis_response(),
            state_snapshot=first["state_snapshot"],
        )
        log_path = (
            project_config.project_root / ".crucible" / "autonomous_sessions" / f"{sid}.jsonl"
        )
        assert log_path.exists()
        lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l]
        events = [line for line in lines]
        assert any('"event": "started"' in e for e in events)
        assert any('"event": "stage_prompted"' in e for e in events)
        assert any('"event": "stage_submitted"' in e for e in events)


# ---------------------------------------------------------------------------
# Doom-loop detection
# ---------------------------------------------------------------------------


class TestDoomLoop:
    def test_repeated_fingerprint_aborts(self, project_config, monkeypatch):
        """Force every prompt build to return the same fingerprint —
        after _DOOM_LOOP_WINDOW (5) stage prompts the session errors.

        Patches ``_fingerprint`` so any sequence of build_prompt calls trips
        the guard. Drives submits until either the loop trips (expected) or
        a hard ceiling is reached (test failure)."""
        monkeypatch.setattr(autos, "_fingerprint", lambda system, user: "STUCK")

        first = autos.action_start(project_config, iterations=99, tier="proxy")
        sid = first["session_id"]
        latest_prompt = first

        with pytest.raises(DoomLoopDetected):
            for _ in range(10):
                stage = latest_prompt["stage"]
                response = (
                    _canned_hypothesis_response()
                    if stage == AutonomousSession.STAGE_HYPOTHESIS
                    else _canned_reflection_response()
                )
                result = autos.action_submit(
                    project_config,
                    session_id=sid,
                    response=response,
                    state_snapshot=latest_prompt["state_snapshot"],
                )
                if result["next_prompt"] is None:
                    break
                latest_prompt = result["next_prompt"]
