"""Tests for the tool_router MCP — Phase 2.1.

Each decision-tree branch in recommend_next_action gets at least one
test. Inputs are controlled by monkeypatching the module-level loader
helpers, so we don't need to spin up real fleet/queue/state files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crucible.mcp import router


class _FakeConfig:
    """Minimal ProjectConfig stand-in for router tests."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.nodes_file = "nodes.json"
        self.research_state_file = "research_state.jsonl"


@pytest.fixture
def fake_config(tmp_path: Path):
    return _FakeConfig(tmp_path)


def _patch(monkeypatch, **kwargs) -> None:
    """Patch named router._load_* helpers with constant returns."""
    for name, value in kwargs.items():
        if callable(value):
            monkeypatch.setattr(router, name, value)
        else:
            monkeypatch.setattr(router, name, lambda *a, _v=value, **k: _v)
    # Always silence the orphan probe so tests don't poke RunPod.
    if "_find_active_session" not in kwargs:
        monkeypatch.setattr(router, "_find_active_session", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Bootstrap chain
# ---------------------------------------------------------------------------


class TestBootstrapChain:
    def test_no_pods_recommends_provision(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": False, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 0.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "provision_nodes"
        assert "No pods" in out["rationale"]
        assert out["state"]["nodes"]["total"] == 0

    def test_pods_no_ssh_recommends_refresh(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[
                {"name": "n1", "state": "running", "ssh_host": None,
                 "env_ready": False, "dataset_ready": False},
            ],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 0.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "fleet_refresh"

    def test_pods_unbootstrapped_recommends_bootstrap(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[
                {"name": "n1", "state": "running",
                 "ssh_host": "1.2.3.4", "env_ready": False,
                 "dataset_ready": False},
            ],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 0.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "bootstrap_nodes"


# ---------------------------------------------------------------------------
# Dispatch / collect
# ---------------------------------------------------------------------------


class TestWorkChain:
    def _ready_node(self):
        return {"name": "n1", "state": "running",
                "ssh_host": "1.2.3.4", "env_ready": True,
                "dataset_ready": True}

    def test_queued_with_idle_pod_recommends_dispatch(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[self._ready_node()],
            _load_queue=[
                {"lease_state": "queued"},
                {"lease_state": "queued"},
            ],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "dispatch_experiments"
        assert out["state"]["queue"]["queued"] == 2
        assert out["state"]["nodes"]["ready"] == 1

    def test_running_experiment_recommends_poll(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[self._ready_node()],
            _load_queue=[{"lease_state": "running"}],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "get_fleet_status"
        assert "running" in out["rationale"].lower()

    def test_finished_uncollected_recommends_collect(self, monkeypatch, fake_config):
        # Queue rows use ``lease_state`` (not ``status``) — Phase 2.1
        # review fix. 3 finished in queue, 1 in local results → collect
        # the other 2.
        _patch(
            monkeypatch,
            _load_nodes=[self._ready_node()],
            _load_queue=[
                {"lease_state": "finished"},
                {"lease_state": "finished"},
                {"lease_state": "completed"},
            ],
            _load_completed_count=1,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "collect_results"
        assert out["state"]["queue"]["finished"] == 3

    def test_partial_fleet_with_ready_pods_dispatches(self, monkeypatch, fake_config):
        """Phase 2.1 review fix: mixed-state fleet (some ready, some
        unbootstrapped) should still dispatch to the ready pods, not
        regress to the bootstrap recommendation."""
        _patch(
            monkeypatch,
            _load_nodes=[
                self._ready_node(),
                self._ready_node(),
                {"name": "n3", "state": "running",
                 "ssh_host": "5.6.7.8", "env_ready": False,
                 "dataset_ready": False},
            ],
            _load_queue=[{"lease_state": "queued"}],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "dispatch_experiments"
        assert out["state"]["nodes"]["ready"] == 2
        assert out["state"]["nodes"]["unbootstrapped"] == 1


# ---------------------------------------------------------------------------
# Hypothesis / batch / reflect
# ---------------------------------------------------------------------------


class TestResearchChain:
    def _ready_node(self):
        return {"name": "n1", "state": "running",
                "ssh_host": "1.2.3.4", "env_ready": True,
                "dataset_ready": True}

    def test_pending_hypotheses_empty_queue_recommends_enqueue(
        self, monkeypatch, fake_config
    ):
        _patch(
            monkeypatch,
            _load_nodes=[self._ready_node()],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={
                "available": True,
                "hypotheses": [{"name": "h1", "status": "pending"}],
                "pending_count": 1, "history_count": 0,
                "budget_remaining": 1.0,
            },
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "design_enqueue_batch"
        assert any(a["tool"] == "design_batch_from_hypotheses"
                   for a in out["alternatives"])

    def test_local_completion_without_pods_recommends_leaderboard(
        self, monkeypatch, fake_config
    ):
        """5-minute quickstart path: ran a local smoke experiment with
        no pods provisioned. Router must NOT regress to provision_nodes
        — completed results are the actionable state."""
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=2,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "get_leaderboard"
        assert out["state"]["nodes"]["total"] == 0
        assert out["state"]["completed_experiments"] == 2

    def test_completed_with_no_pending_recommends_leaderboard(
        self, monkeypatch, fake_config
    ):
        _patch(
            monkeypatch,
            _load_nodes=[self._ready_node()],
            _load_queue=[],
            _load_completed_count=5,
            _load_research_state={
                "available": True, "hypotheses": [],
                "pending_count": 0, "history_count": 2,
                "budget_remaining": 1.0,
            },
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "get_leaderboard"
        # Reflection should appear in alternatives.
        alt_tools = [a["tool"] for a in out["alternatives"]]
        assert "research_request_prompt" in alt_tools

    def test_empty_project_recommends_first_hypothesis(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[self._ready_node()],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={
                "available": False, "hypotheses": [],
                "pending_count": 0, "history_count": 0,
                "budget_remaining": 0.0,
            },
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "research_request_prompt"
        assert "hypothesis" in out["rationale"].lower()


# ---------------------------------------------------------------------------
# Active session takes priority
# ---------------------------------------------------------------------------


class TestActiveSession:
    def test_active_autonomous_session_recommends_loop_tool(
        self, monkeypatch, fake_config
    ):
        monkeypatch.setattr(
            router, "_find_active_session",
            lambda *a, **k: {
                "kind": "autonomous", "session_id": "s-abc",
                "stage": "hypothesis", "status": "running",
            },
        )
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
            _find_active_session=lambda *a, **k: {
                "kind": "autonomous", "session_id": "s-abc",
                "stage": "hypothesis", "status": "running",
            },
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "autonomous_research_loop"
        assert "s-abc" in out["rationale"]

    def test_active_tree_session_maps_to_tree_loop(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
            _find_active_session=lambda *a, **k: {
                "kind": "tree", "session_id": "t-xyz",
                "stage": "external_dispatch", "status": "running",
            },
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "tree_autonomous_loop"
        # Wait-stage hint should reach the rationale.
        assert "status" in out["rationale"] or "wait" in out["rationale"].lower()

    def test_active_harness_session_maps_to_harness_loop(
        self, monkeypatch, fake_config
    ):
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
            _find_active_session=lambda *a, **k: {
                "kind": "harness", "session_id": "h-001",
                "stage": "proposal", "status": "running",
            },
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "harness_autonomous_loop"


# ---------------------------------------------------------------------------
# Response shape contract
# ---------------------------------------------------------------------------


class TestResponseShape:
    def test_response_has_required_keys(self, monkeypatch, fake_config):
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 0.0},
        )
        out = router.recommend_next_action(fake_config)
        for key in ("recommended_tool", "rationale", "alternatives", "state"):
            assert key in out, f"Missing required key {key!r}"
        for alt in out["alternatives"]:
            assert "tool" in alt and "rationale" in alt
        state = out["state"]
        for k in ("nodes", "queue", "completed_experiments",
                  "hypotheses_pending", "active_session", "orphans_present"):
            assert k in state, f"State missing {k!r}"

    def test_mcp_dispatch_routes_to_router(self, monkeypatch, fake_config):
        from crucible.mcp.tools import TOOL_DISPATCH
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: fake_config)
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 0.0},
        )
        handler = TOOL_DISPATCH["tool_router"]
        out = handler({})
        assert out["recommended_tool"] == "provision_nodes"


# ---------------------------------------------------------------------------
# Orphan-probe opt-in (Phase 2.1 review fix)
# ---------------------------------------------------------------------------


class TestOrphanProbe:
    def test_orphan_probe_skipped_by_default(self, monkeypatch, fake_config):
        """check_orphans defaults to False so the live RunPod GraphQL
        round-trip in cleanup_orphans never fires unless caller asks."""
        called = {"n": 0}

        def boom(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("FleetManager should not be touched by default")

        monkeypatch.setattr("crucible.fleet.manager.FleetManager", boom)
        _patch(
            monkeypatch,
            _load_nodes=[],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 0.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["state"]["orphans_present"] is False
        assert called["n"] == 0

    def test_orphan_probe_runs_when_opted_in(self, monkeypatch, fake_config):
        class _StubFM:
            def __init__(self, _config):
                pass
            def cleanup_orphans(self, *, destroy, include_legacy):
                return {"tagged_orphans": [{"pod_id": "abc"}]}

        monkeypatch.setattr("crucible.fleet.manager.FleetManager", _StubFM)
        _patch(
            monkeypatch,
            _load_nodes=[{"name": "n1", "state": "running",
                          "ssh_host": "1.2.3.4", "env_ready": True,
                          "dataset_ready": True}],
            _load_queue=[],
            _load_completed_count=0,
            _load_research_state={"available": True, "hypotheses": [],
                                  "pending_count": 0, "history_count": 0,
                                  "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config, check_orphans=True)
        assert out["state"]["orphans_present"] is True
        assert out["recommended_tool"] == "cleanup_orphans"


# ---------------------------------------------------------------------------
# Session-id unpacking correctness (Phase 2.1 review fix)
# ---------------------------------------------------------------------------


class TestFindActiveSessionUnpack:
    def test_autonomous_session_stage_uses_current_stage_field(
        self, monkeypatch, fake_config, tmp_path
    ):
        """Regression: live MCP test of autonomous_research_loop showed
        active_session.stage == null because the router read
        ``data.get('stage')`` but AutonomousSession persists the field
        as ``current_stage``. Three sibling drivers diverge on this:
        autonomous uses ``current_stage``, harness uses ``stage``, tree
        has no stage field. Router must read both keys."""
        sessions_dir = tmp_path / ".crucible" / "autonomous_sessions"
        sessions_dir.mkdir(parents=True)
        import yaml as _yaml
        sid = "session-uuid-abc123"
        (sessions_dir / f"{sid}.yaml").write_text(
            _yaml.safe_dump({
                "session_id": sid,
                "schema_version": 1,
                "status": "running",
                "current_stage": "hypothesis",  # autonomous-session key
                "tree_name": None,
                "iterations_planned": 3,
                "iterations_completed": 0,
                "current_iteration": 0,
                "started_at": "2026-05-16T00:00:00Z",
                "updated_at": "2026-05-16T01:00:00Z",
                "tier": "smoke",
                "focus_family": None,
                "project_name": "demo",
                "budget_usd": None,
                "budget_spent_usd": 0.0,
                "with_literature": False,
                "literature_k": 5,
                "last_state_snapshot": None,
                "recent_fingerprints": [],
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(router, "_load_nodes", lambda *a, **k: [])
        monkeypatch.setattr(router, "_load_queue", lambda *a, **k: [])
        monkeypatch.setattr(router, "_load_completed_count", lambda *a, **k: 0)
        monkeypatch.setattr(
            router, "_load_research_state",
            lambda *a, **k: {"available": True, "hypotheses": [],
                             "pending_count": 0, "history_count": 0,
                             "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["state"]["active_session"]["stage"] == "hypothesis", (
            f"stage should be 'hypothesis', got {out['state']['active_session']['stage']!r}"
        )

    def test_session_id_in_response_is_actual_id_not_timestamp(
        self, monkeypatch, fake_config, tmp_path
    ):
        """Regression: _find_active_yamls returns (updated_at, session_id,
        data); the previous unpack swapped the first two so the
        response surfaced the timestamp as session_id."""
        # Build a real session yaml on disk so _find_active_session
        # exercises the actual 3-tuple unpacking path.
        sessions_dir = tmp_path / ".crucible" / "autonomous_sessions"
        sessions_dir.mkdir(parents=True)
        import yaml as _yaml
        sid = "abcd1234-real-session-id"
        (sessions_dir / f"{sid}.yaml").write_text(
            _yaml.safe_dump({
                "session_id": sid,
                "schema_version": 1,
                "stage": "hypothesis",
                "status": "running",
                "tree_name": None,
                "iterations_planned": 3,
                "iterations_completed": 0,
                "current_iteration": 0,
                "started_at": "2026-05-15T00:00:00Z",
                "updated_at": "2026-05-15T01:00:00Z",
                "tier": "proxy",
                "focus_family": None,
                "project_name": "demo",
                "budget_usd": None,
                "budget_spent_usd": 0.0,
                "with_literature": False,
                "literature_k": 5,
                "last_state_snapshot": None,
                "recent_fingerprints": [],
            }),
            encoding="utf-8",
        )
        # Patch only the data loaders, NOT _find_active_session, so the
        # real session-scan path runs against the yaml we just wrote.
        monkeypatch.setattr(router, "_load_nodes", lambda *a, **k: [])
        monkeypatch.setattr(router, "_load_queue", lambda *a, **k: [])
        monkeypatch.setattr(router, "_load_completed_count", lambda *a, **k: 0)
        monkeypatch.setattr(
            router, "_load_research_state",
            lambda *a, **k: {"available": True, "hypotheses": [],
                             "pending_count": 0, "history_count": 0,
                             "budget_remaining": 1.0},
        )
        out = router.recommend_next_action(fake_config)
        assert out["recommended_tool"] == "autonomous_research_loop"
        # The crucial assertion: session_id is the real one, not a timestamp.
        assert out["state"]["active_session"]["session_id"] == sid
        assert sid in out["rationale"]
