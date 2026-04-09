"""Tests for bootstrap state verification (Phase 2b).

Validates the _record_step helper + bootstrap_state_summary: nodes get a
per-step tracking dict, required failures re-raise, optional failures are
logged and swallowed, and the aggregate summary reflects what actually ran.
"""
from __future__ import annotations

import pytest

from crucible.fleet.bootstrap import (
    _record_step,
    bootstrap_state_summary,
)


# ---------------------------------------------------------------------------
# _record_step
# ---------------------------------------------------------------------------


class TestRecordStep:
    def test_success_records_ok_status(self):
        node: dict = {"name": "test"}
        result = _record_step(node, "step_a", lambda: "value")
        assert result == "value"
        assert node["bootstrap_steps"]["step_a"]["status"] == "ok"
        assert node["bootstrap_steps"]["step_a"]["error"] is None
        assert node["bootstrap_steps"]["step_a"]["required"] is True

    def test_required_failure_reraises(self):
        node: dict = {"name": "test"}

        def boom() -> None:
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            _record_step(node, "step_b", boom)

        # Failure still recorded even though it re-raised
        assert node["bootstrap_steps"]["step_b"]["status"] == "failed"
        assert "kaboom" in node["bootstrap_steps"]["step_b"]["error"]
        assert node["bootstrap_steps"]["step_b"]["required"] is True

    def test_optional_failure_is_swallowed(self):
        node: dict = {"name": "test"}

        def flaky() -> None:
            raise RuntimeError("probe is optional")

        # Must NOT raise
        result = _record_step(node, "probe", flaky, required=False)
        assert result is None
        assert node["bootstrap_steps"]["probe"]["status"] == "failed"
        assert node["bootstrap_steps"]["probe"]["required"] is False

    def test_multiple_steps_accumulate(self):
        node: dict = {"name": "test"}
        _record_step(node, "a", lambda: 1)
        _record_step(node, "b", lambda: 2)
        _record_step(node, "c", lambda: 3)
        assert set(node["bootstrap_steps"].keys()) == {"a", "b", "c"}
        assert all(
            node["bootstrap_steps"][k]["status"] == "ok" for k in ("a", "b", "c")
        )

    def test_step_has_timestamps(self):
        node: dict = {"name": "test"}
        _record_step(node, "ts", lambda: None)
        info = node["bootstrap_steps"]["ts"]
        assert info["started_at"]
        assert info["finished_at"]


# ---------------------------------------------------------------------------
# bootstrap_state_summary
# ---------------------------------------------------------------------------


class TestBootstrapStateSummary:
    def test_empty_node_returns_trivially_ok(self):
        summary = bootstrap_state_summary({"name": "empty"})
        assert summary["total"] == 0
        assert summary["ok"] == 0
        assert summary["failed"] == []
        assert summary["required_failed"] == 0
        assert summary["all_required_ok"] is True

    def test_all_ok(self):
        node: dict = {"name": "n"}
        _record_step(node, "a", lambda: None)
        _record_step(node, "b", lambda: None)
        summary = bootstrap_state_summary(node)
        assert summary["total"] == 2
        assert summary["ok"] == 2
        assert summary["failed"] == []
        assert summary["all_required_ok"] is True

    def test_required_failure_flips_aggregate(self):
        node: dict = {"name": "n"}
        _record_step(node, "a", lambda: None)
        with pytest.raises(ValueError):
            _record_step(node, "b", lambda: (_ for _ in ()).throw(ValueError("x")))
        summary = bootstrap_state_summary(node)
        assert summary["total"] == 2
        assert summary["ok"] == 1
        assert summary["required_failed"] == 1
        assert summary["all_required_ok"] is False
        assert len(summary["failed"]) == 1
        assert summary["failed"][0]["step"] == "b"

    def test_optional_failure_does_not_flip(self):
        node: dict = {"name": "n"}
        _record_step(node, "req", lambda: None)
        _record_step(
            node, "opt",
            lambda: (_ for _ in ()).throw(RuntimeError("meh")),
            required=False,
        )
        summary = bootstrap_state_summary(node)
        assert summary["total"] == 2
        assert summary["ok"] == 1
        assert summary["required_failed"] == 0
        assert summary["all_required_ok"] is True
        assert len(summary["failed"]) == 1
        assert summary["failed"][0]["required"] is False

    def test_mixed_required_and_optional(self):
        node: dict = {"name": "n"}
        _record_step(node, "a_ok", lambda: None)
        _record_step(
            node, "b_optional_fail",
            lambda: (_ for _ in ()).throw(RuntimeError("flake")),
            required=False,
        )
        _record_step(node, "c_ok", lambda: None)
        with pytest.raises(ValueError):
            _record_step(
                node, "d_required_fail",
                lambda: (_ for _ in ()).throw(ValueError("halt")),
            )
        summary = bootstrap_state_summary(node)
        assert summary["total"] == 4
        assert summary["ok"] == 2
        assert summary["required_failed"] == 1
        assert summary["all_required_ok"] is False
        failed_steps = {f["step"]: f["required"] for f in summary["failed"]}
        assert failed_steps == {"b_optional_fail": False, "d_required_fail": True}
