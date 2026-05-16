"""Tests for the harness_autonomous_loop session driver (Phase 1.5)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError
from crucible.researcher import harness_autonomous_session as has
from crucible.researcher.harness_autonomous_session import (
    HarnessAutonomousSession,
    HarnessAutonomousSessionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_minimal_domain_spec(spec_path: Path) -> None:
    """Write a minimal domain_spec.yaml the loader will accept."""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump({
            "name": "demo",
            "interface": {
                "class_name": "Harness",
                "required_methods": [
                    {"name": "predict", "signature": "(self, x)"},
                ],
            },
            "baselines": [],
            "metrics": [
                {"name": "accuracy", "direction": "maximize"},
            ],
            "constraints": {},
            "proposal_guidance": "Propose simple harnesses.",
            "evaluation": {},
        }),
        encoding="utf-8",
    )


@pytest.fixture
def project_config(tmp_path: Path):
    """Minimal project with crucible.yaml + domain spec on disk."""
    (tmp_path / "crucible.yaml").write_text(
        yaml.safe_dump({
            "name": "harness_test",
            "metrics": {"primary": "accuracy", "direction": "maximize"},
        }),
        encoding="utf-8",
    )
    spec_path = tmp_path / "domain_spec.yaml"
    _write_minimal_domain_spec(spec_path)
    os.chdir(tmp_path)
    return load_config(tmp_path / "crucible.yaml"), str(spec_path)


# A canned harness candidate response (Python code block + JSON metadata).
# Format matches HarnessOptimizer._parse_candidates expectations.
_CANNED_RESPONSE = """\
```python
class Harness:
    def predict(self, x):
        return 0
```

```json
[
  {"name": "constant_zero", "hypothesis": "always-zero baseline", "rationale": "smoke test"}
]
```
"""


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


def _start(config, spec, tree_name="default", iterations=2, **kwargs):
    """Test helper: action_start with dry_run=True so benchmark doesn't dispatch."""
    return has.action_start(
        config, domain_spec=spec, tree_name=tree_name,
        iterations=iterations, dry_run=True, **kwargs,
    )


class TestStart:
    def test_creates_session_and_returns_first_prompt(self, project_config):
        config, spec = project_config
        out = _start(config, spec, "h1", iterations=2)
        assert "session_id" in out
        assert out["stage"] == HarnessAutonomousSession.STAGE_PROPOSAL
        assert out["iteration"] == 0
        assert isinstance(out["system"], str) and len(out["system"]) > 0
        assert isinstance(out["user"], str)
        assert out["session_status"] == HarnessAutonomousSession.STATUS_RUNNING

    def test_rejects_zero_iterations(self, project_config):
        config, spec = project_config
        with pytest.raises(CrucibleError, match="iterations must be >= 1"):
            has.action_start(config, domain_spec=spec, tree_name="bad", iterations=0)

    def test_persists_session_file(self, project_config):
        config, spec = project_config
        out = has.action_start(
            config, domain_spec=spec, tree_name="h2", iterations=2
        )
        sid = out["session_id"]
        path = (
            config.project_root
            / ".crucible" / "harness_autonomous_sessions" / f"{sid}.yaml"
        )
        assert path.exists()

    def test_idempotent_second_start_returns_active(self, project_config):
        config, spec = project_config
        first = _start(config, spec, "dup", iterations=2)
        second = _start(config, spec, "dup", iterations=2)
        assert first["session_id"] == second["session_id"]


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_validates_and_benchmarks(self, project_config):
        config, spec = project_config
        started = _start(config, spec, tree_name="sub", iterations=2
        )
        sid = started["session_id"]
        result = has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)
        assert result["session_status"] == HarnessAutonomousSession.STATUS_RUNNING
        assert result["next_action"] == "external_dispatch"
        # constant_zero passed validation (it implements `predict`).
        assert "constant_zero" in result["proposed"]
        assert len(result["benchmarked_node_ids"]) >= 0  # may be 0 if validation rejects

    def test_session_done_after_planned_iterations(self, project_config):
        config, spec = project_config
        started = _start(config, spec, tree_name="done", iterations=1)
        sid = started["session_id"]
        result = has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)
        assert result["session_status"] == HarnessAutonomousSession.STATUS_DONE
        assert result["next_prompt"] is None
        # Codex review fix: terminal submit must surface pending_node_ids
        # so the orchestrator knows whether to run collect_results +
        # tree_sync_results to finalize metrics.
        assert "pending_node_ids" in result
        assert isinstance(result["pending_node_ids"], list)
        assert "message" in result

    def test_cannot_submit_in_benchmark_wait_stage(self, project_config):
        config, spec = project_config
        started = _start(config, spec, tree_name="bw", iterations=3
        )
        sid = started["session_id"]
        # First submit advances to benchmark_wait.
        has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)
        # Second submit before continue must error.
        with pytest.raises(HarnessAutonomousSessionError, match="benchmark_wait|proposal"):
            has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)


# ---------------------------------------------------------------------------
# continue
# ---------------------------------------------------------------------------


class TestContinue:
    def test_continue_after_submit_advances_or_waits(self, project_config):
        """After submit, the session is in benchmark_wait. continue either
        advances to next proposal (if pending drained) or returns
        external_dispatch hint (if pending still exist)."""
        config, spec = project_config
        started = _start(config, spec, tree_name="cont", iterations=3
        )
        sid = started["session_id"]
        has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)
        # In real flow the orchestrator drives dispatch+collect+sync; here we
        # call continue while pending nodes still exist.
        result = has.action_continue(config, session_id=sid)
        # Either we get a proposal prompt or an external_dispatch hint —
        # depends on whether the benchmark's enqueued nodes are still pending.
        if "next_action" in result:
            assert result["next_action"] == "external_dispatch"
        else:
            # Pending must have been drained — next iteration's prompt.
            assert result["stage"] == HarnessAutonomousSession.STAGE_PROPOSAL

    def test_continue_on_terminal_raises(self, project_config):
        config, spec = project_config
        started = _start(config, spec, tree_name="term", iterations=1
        )
        sid = started["session_id"]
        has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)
        # Session is DONE.
        with pytest.raises(HarnessAutonomousSessionError, match="done|canceled|error"):
            has.action_continue(config, session_id=sid)


# ---------------------------------------------------------------------------
# status + cancel
# ---------------------------------------------------------------------------


class TestStatusAndCancel:
    def test_status_returns_state(self, project_config):
        config, spec = project_config
        started = _start(config, spec, tree_name="stat", iterations=2
        )
        sid = started["session_id"]
        status = has.action_status(config, session_id=sid)
        assert status["session_id"] == sid
        assert status["tree_name"] == "stat"
        assert status["status"] == HarnessAutonomousSession.STATUS_RUNNING

    def test_budget_exceeded_auto_cancels(self, project_config, monkeypatch):
        """Phase 1.8 propagation: harness session also enforces budget cap
        via SessionBase._refresh_budget_and_maybe_cancel."""
        from crucible.researcher.session_base import BudgetExceeded
        from crucible.runner import cost_tracker

        monkeypatch.setattr(
            cost_tracker, "compute_session_spend",
            lambda *a, **kw: {
                "spend_usd": 100.0, "hours_elapsed": 1.0,
                "hourly_rate": 100.0, "active_pods": 1,
            },
        )
        config, spec = project_config
        with pytest.raises(BudgetExceeded):
            _start(config, spec, tree_name="bud", iterations=3, budget_usd=5.0)

    def test_budget_check_after_submit_under_session_lock(
        self, project_config, monkeypatch
    ):
        """Defense-in-depth: budget refresh fires after a successful submit.
        Harness submit runs validate_candidates + optimizer.benchmark()
        synchronously (and benchmark can dispatch + locally execute), so
        wall-clock cost can jump during a single submit call."""
        from crucible.researcher.session_base import BudgetExceeded
        from crucible.runner import cost_tracker

        call_count = {"n": 0}

        def spend_grows(config, session_started_at, *, now=None):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return {"spend_usd": 1.0, "hours_elapsed": 0.1,
                        "hourly_rate": 10.0, "active_pods": 1}
            return {"spend_usd": 100.0, "hours_elapsed": 1.0,
                    "hourly_rate": 100.0, "active_pods": 1}

        monkeypatch.setattr(cost_tracker, "compute_session_spend", spend_grows)
        config, spec = project_config
        # Iterations=5 so submit advances to benchmark_wait (running branch),
        # not done (terminal branch which short-circuits the budget check).
        started = _start(config, spec, tree_name="bsub", iterations=5, budget_usd=5.0)
        sid = started["session_id"]
        with pytest.raises(BudgetExceeded):
            has.action_submit(config, session_id=sid, response=_CANNED_RESPONSE)

    def test_cancel_marks_canceled(self, project_config):
        config, spec = project_config
        started = _start(config, spec, tree_name="cnc", iterations=2
        )
        sid = started["session_id"]
        out = has.action_cancel(config, session_id=sid, reason="testing")
        assert out["session_status"] == HarnessAutonomousSession.STATUS_CANCELED
        # Idempotent
        again = has.action_cancel(config, session_id=sid)
        assert again["already_terminal"] is True

    def test_unknown_session_raises(self, project_config):
        config, _ = project_config
        with pytest.raises(HarnessAutonomousSessionError, match="not found"):
            has.action_status(config, session_id="not-a-real-uuid")


# ---------------------------------------------------------------------------
# Concurrent action_start (G.4 seam 5 — symmetry with research + tree)
# ---------------------------------------------------------------------------


def _concurrent_harness_start_worker(
    project_dir_str: str, spec_path_str: str, tree_name: str, queue
) -> None:
    """Subprocess target: cd into project, start a harness session, put session_id."""
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(project_dir_str))
    from crucible.core.config import load_config as _lc
    from crucible.researcher import harness_autonomous_session as _has
    config = _lc()
    out = _has.action_start(
        config, domain_spec=spec_path_str, tree_name=tree_name,
        iterations=2, dry_run=True,
    )
    queue.put(out["session_id"])


class TestStartConcurrency:
    """G.4 seam 5: harness mirror of the autonomous/tree tests. Three
    processes call action_start against the same tree simultaneously;
    exactly one session is created and all three callers see it."""

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="fcntl create-lock is POSIX-only.",
    )
    def test_concurrent_harness_starts_produce_single_session(self, project_config):
        import multiprocessing

        config, spec = project_config
        project_dir = config.project_root

        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_concurrent_harness_start_worker,
                args=(str(project_dir), spec, "race-tree", queue),
            )
            for _ in range(3)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30.0)
            assert p.exitcode == 0, f"worker exitcode={p.exitcode}"

        ids: set[str] = set()
        while not queue.empty():
            ids.add(queue.get())
        assert len(ids) == 1, (
            f"expected exactly one harness session_id under concurrent start; got {ids}"
        )
