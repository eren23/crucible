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

    def test_create_lock_file_exists(self, project_config):
        """Phase 1.1 review fix: a project-level create lock must exist so
        concurrent action_start calls cannot create two distinct sessions."""
        autos.action_start(project_config, iterations=2, tier="proxy")
        lock_path = (
            project_config.project_root
            / ".crucible" / "autonomous_sessions" / ".create.lock"
        )
        assert lock_path.exists()

    def test_judge_separation_enforced_at_start(self, project_config):
        """Phase 1.9: AutonomousSession.create calls panel.assert_separated()
        before minting a session, so misconfigured judges fail before any pod
        time is consumed."""
        from crucible.core.config import JudgeConfig, JudgePanel
        from crucible.core.errors import ConfigError

        # Force a misconfiguration: same model, same family.
        project_config.judges = JudgePanel(
            reward_judge=JudgeConfig(model="claude-3", family="claude"),
            eval_judge=JudgeConfig(model="claude-3", family="claude"),
            enforce_separation=True,
        )
        with pytest.raises(ConfigError):
            autos.action_start(project_config, iterations=2, tier="proxy")

    def test_judge_separation_skipped_when_unconfigured(self, project_config):
        """An empty (unconfigured) JudgePanel must NOT block session start —
        the contract is opt-in via populated model strings."""
        from crucible.core.config import JudgePanel
        project_config.judges = JudgePanel()  # empty model strings
        out = autos.action_start(project_config, iterations=2, tier="proxy")
        assert "session_id" in out

    def test_project_name_persisted_in_state(self, project_config):
        """Phase 1.7 review fix: project_name was a dead key in the literature
        query fallback because it was never stored in state_data. The fix
        captures config.name at create-time."""
        out = autos.action_start(project_config, iterations=2, tier="proxy")
        status = autos.action_status(project_config, session_id=out["session_id"])
        # Should be the config's project name (or empty string for empty config).
        assert "project_name" in status

    def test_budget_exceeded_auto_cancels_session(
        self, project_config, monkeypatch
    ):
        """Phase 1.8: when wall-clock × declared rate exceeds budget_usd,
        the next prompt build auto-cancels the session and raises
        BudgetExceeded. State persists last_error + budget_spent_usd."""
        from crucible.researcher.autonomous_session import BudgetExceeded
        from crucible.runner import cost_tracker

        # Mock the spend computation to return a value already over the cap.
        def over_budget(config, session_started_at, *, now=None):
            return {
                "spend_usd": 100.0,
                "hours_elapsed": 1.0,
                "hourly_rate": 100.0,
                "active_pods": 1,
            }
        monkeypatch.setattr(cost_tracker, "compute_session_spend", over_budget)
        # The autonomous_session module imports compute_session_spend lazily
        # inside _refresh_budget_and_maybe_cancel, so monkeypatching the
        # source module is sufficient.

        with pytest.raises(BudgetExceeded, match="budget exceeded"):
            autos.action_start(
                project_config, iterations=5, tier="proxy", budget_usd=5.0,
            )

    def test_budget_check_after_submit_under_session_lock(
        self, project_config, monkeypatch
    ):
        """Phase 1.8 fix: budget check also fires after a successful submit
        (matches the docstring claim 'each prompt build AND each successful
        submit'). The check runs while the session lock is still held so
        cancel-on-overrun is atomic vs concurrent reads.

        Simulate: start with a high budget (passes initial check), then
        flip the cost spend to over-budget before submit runs. Submit
        must mark the session canceled."""
        from crucible.researcher.autonomous_session import BudgetExceeded
        from crucible.runner import cost_tracker

        # First spend call (during start's build_prompt) returns under-budget.
        # Subsequent calls return over-budget so the post-submit refresh
        # cancels the session.
        call_count = {"n": 0}
        def spend_grows(config, session_started_at, *, now=None):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return {"spend_usd": 1.0, "hours_elapsed": 0.1,
                        "hourly_rate": 10.0, "active_pods": 1}
            return {"spend_usd": 100.0, "hours_elapsed": 1.0,
                    "hourly_rate": 100.0, "active_pods": 1}
        monkeypatch.setattr(cost_tracker, "compute_session_spend", spend_grows)

        started = autos.action_start(
            project_config, iterations=5, tier="proxy", budget_usd=5.0,
        )
        sid = started["session_id"]

        # Now submit — the post-apply budget check should trip BudgetExceeded.
        resp = {"hypotheses": [{"hypothesis": "h", "name": "h",
                                "expected_impact": 0.01, "confidence": 0.5,
                                "config": {"MODEL_FAMILY": "baseline"},
                                "rationale": "x", "family": "baseline"}]}
        with pytest.raises(BudgetExceeded):
            autos.action_submit(
                project_config, session_id=sid, response=resp,
                state_snapshot=started["state_snapshot"],
            )

    def test_budget_none_skips_check(self, project_config, monkeypatch):
        """budget_usd=None opts out of budget enforcement entirely."""
        from crucible.runner import cost_tracker

        called = {"n": 0}
        def tracking(config, session_started_at, *, now=None):
            called["n"] += 1
            return {"spend_usd": 0.0, "hours_elapsed": 0.0, "hourly_rate": 0.0, "active_pods": 0}
        monkeypatch.setattr(cost_tracker, "compute_session_spend", tracking)

        autos.action_start(project_config, iterations=2, tier="proxy")
        # Default budget_usd is None — compute_session_spend should never
        # be invoked.
        assert called["n"] == 0

    def test_with_literature_doesnt_crash_on_network_failure(
        self, project_config, monkeypatch
    ):
        """Phase 1.7 contract: literature pre-injection is best-effort.
        Even if search_papers blows up (network down, parse error),
        build_prompt must NOT crash — literature_context degrades to
        empty string and the prompt is still returned."""
        from crucible.researcher import literature

        def boom(*args, **kwargs):
            raise RuntimeError("simulated network down")

        monkeypatch.setattr(literature, "search_papers", boom)

        out = autos.action_start(
            project_config, iterations=2, tier="proxy",
            with_literature=True, literature_k=3,
        )
        # No crash — prompt returned normally.
        assert "session_id" in out
        assert "system" in out
        assert "user" in out


class TestStartConcurrency:
    """Phase 1.1 review fix: prevent two concurrent action_start from creating
    distinct sessions. Uses multiprocessing to simulate real cross-process race."""

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="fcntl locks are POSIX-only; create-lock degrades to no-op on Windows.",
    )
    def test_concurrent_starts_produce_single_session(self, project_dir):
        import multiprocessing

        # The project_config fixture changes cwd; this test needs to do that
        # inside the child processes too.
        os.chdir(project_dir)
        (project_dir / "program.md").write_text("Minimise val_loss.", encoding="utf-8")

        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=_concurrent_start_worker, args=(str(project_dir), queue))
            for _ in range(3)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=20.0)
            assert p.exitcode == 0, f"worker exited with {p.exitcode}"

        ids: set[str] = set()
        while not queue.empty():
            ids.add(queue.get())
        assert len(ids) == 1, (
            f"expected exactly one session_id across concurrent starts, got {ids}"
        )


def _concurrent_start_worker(project_dir_str: str, queue) -> None:
    """Subprocess target: cd into project, start a session, put session_id."""
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(project_dir_str))
    from crucible.core.config import load_config as _lc
    from crucible.researcher import autonomous_session as _autos
    config = _lc()
    out = _autos.action_start(config, iterations=2, tier="proxy")
    queue.put(out["session_id"])


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

    def test_explicit_stale_snapshot_rejected_via_mcp_dispatch(
        self, project_config, monkeypatch
    ):
        """G.4 seam 7: full MCP-style flow — action_start returns a
        snapshot, peer mutates state, MCP dispatcher passes the stale
        snapshot through autonomous_research_loop(action="submit").
        Must raise StaleSubmitError so the orchestrator can retry.

        Earlier coverage tested the snapshot fallback path (no explicit
        snapshot) and the Python-API path. This test exercises the
        explicit MCP arg + dispatcher path that production callers
        actually use."""
        from crucible.mcp.tools import TOOL_DISPATCH
        from crucible.researcher.state import ResearchState

        monkeypatch.setattr(
            "crucible.mcp.tools._get_config", lambda: project_config
        )
        loop = TOOL_DISPATCH["autonomous_research_loop"]
        started = loop({"action": "start", "iterations": 2, "tier": "proxy"})
        sid = started["session_id"]
        stale = started["state_snapshot"]

        # Peer mutation between start and submit.
        state = ResearchState(
            project_config.project_root / project_config.research_state_file,
            budget_hours=project_config.researcher.budget_hours,
        )
        state.add_finding("peer-finding", confidence=0.7)
        state.save()

        # Submit with the stale snapshot — must surface StaleSubmitError
        # through the dispatcher.
        with pytest.raises(StaleSubmitError):
            loop({
                "action": "submit",
                "session_id": sid,
                "response": _canned_hypothesis_response(),
                "state_snapshot": stale,
            })

    def test_action_submit_falls_back_to_session_snapshot(self, project_config):
        """Phase 1.1 review fix: if caller (e.g., CLI) omits state_snapshot,
        action_submit auto-loads the session's last_state_snapshot so stale-
        submit detection still kicks in. Bypassing it is no longer the default."""
        first = autos.action_start(project_config, iterations=2, tier="proxy")
        sid = first["session_id"]

        # Concurrent process advances state on disk (persists).
        from crucible.researcher.state import ResearchState
        state = ResearchState(
            project_config.project_root / project_config.research_state_file,
            budget_hours=project_config.researcher.budget_hours,
        )
        state.add_finding("peer", confidence=0.7)
        state.save()

        # Caller passes no state_snapshot — action_submit must load the
        # session's last_state_snapshot (captured at start) and detect the
        # mismatch.
        with pytest.raises(StaleSubmitError):
            autos.action_submit(
                project_config,
                session_id=sid,
                response=_canned_hypothesis_response(),
                # No state_snapshot — relies on the auto-load.
            )


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


class TestBudgetSpendDictValidation:
    """G.4 seam 2: cost_tracker.compute_session_spend returning a
    malformed dict used to raise a confusing KeyError inside the budget
    check. Now raises typed BudgetCheckError so the operator sees the
    cost-tracker bug not a fake budget-exceeded."""

    def test_missing_key_raises_typed_error(self, project_config, monkeypatch):
        from crucible.researcher.session_base import BudgetCheckError
        from crucible.runner import cost_tracker

        monkeypatch.setattr(
            cost_tracker, "compute_session_spend",
            lambda *a, **kw: {"spend_usd": 1.0},  # missing hours/rate/pods
        )
        with pytest.raises(BudgetCheckError, match="missing keys"):
            autos.action_start(project_config, iterations=3, tier="proxy",
                               budget_usd=5.0)

    def test_non_numeric_spend_raises_typed_error(
        self, project_config, monkeypatch
    ):
        from crucible.researcher.session_base import BudgetCheckError
        from crucible.runner import cost_tracker

        monkeypatch.setattr(
            cost_tracker, "compute_session_spend",
            lambda *a, **kw: {
                "spend_usd": "not a number",
                "hours_elapsed": 0.1, "hourly_rate": 10.0, "active_pods": 1,
            },
        )
        with pytest.raises(BudgetCheckError, match="expected number"):
            autos.action_start(project_config, iterations=3, tier="proxy",
                               budget_usd=5.0)

    def test_non_dict_return_raises_typed_error(
        self, project_config, monkeypatch
    ):
        from crucible.researcher.session_base import BudgetCheckError
        from crucible.runner import cost_tracker

        monkeypatch.setattr(
            cost_tracker, "compute_session_spend",
            lambda *a, **kw: None,
        )
        with pytest.raises(BudgetCheckError, match="expected dict"):
            autos.action_start(project_config, iterations=3, tier="proxy",
                               budget_usd=5.0)


class TestCorruptSessionYaml:
    """G.4 seam 1: corrupt session yaml used to vanish silently."""

    def test_corrupt_yaml_logs_warning_and_skips(
        self, project_config, capsys
    ):
        from crucible.researcher.autonomous_session import AutonomousSession

        sessions_dir = (
            project_config.project_root / ".crucible" / "autonomous_sessions"
        )
        sessions_dir.mkdir(parents=True, exist_ok=True)
        # Syntactically broken yaml (unclosed bracket + key collision).
        (sessions_dir / "broken-session.yaml").write_text(
            "status: running\n  current_stage: : :\nunclosed:[\n",
            encoding="utf-8",
        )

        results = AutonomousSession._find_active_yamls(project_config)

        # Corrupted yaml is dropped from results.
        names = [sid for _updated_at, sid, _data in results]
        assert "broken-session" not in names

        # log_warn prints to stderr; capsys captures it.
        captured = capsys.readouterr()
        assert "broken-session.yaml" in captured.err
        assert "_find_active_yamls" in captured.err


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
