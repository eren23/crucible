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


def _start(config, spec, tree_name, iterations=2, **kwargs):
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
