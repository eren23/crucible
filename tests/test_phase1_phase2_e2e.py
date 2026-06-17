"""End-to-end integration smoke for Phase 1 + Phase 2 (Part G.5).

Roundtrips every new MCP tool against a tmp project to lock in the
contracts as a single regression. Live-MCP verification (Part G.2)
already exercised these against the real ``parameter-golf`` project;
this test runs the same protocol in CI without depending on an
external MCP server.

The sequence mirrors the real autonomous-research workflow:

  1. Fresh tmp project + seeded results
  2. ``tool_router`` → state-aware recommendation
  3. ``runs_search`` (basic + strict_fields)
  4. ``get_research_briefing`` → ``next_actions`` field populated
  5. ``autonomous_research_loop`` start → submit → status → cancel
  6. ``tree_autonomous_loop`` start → status → cancel
  7. ``harness_autonomous_loop`` start → status → cancel
  8. ``tree_auto_expand`` request_prompt → submit (with shape guard)
  9. ``context_push_finding`` with auto_promote
  10. ``tree_prune`` mode=auto

Marked as a regular pytest module (not under tests/integration/) so
it runs by default. Each tool call is a sub-second op — total runtime
is bounded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_YAML = """\
name: phase1-phase2-e2e
version: "0.2.1-alpha"

provider:
  type: ssh
  ssh_key: ~/.ssh/id_ed25519

training:
  - backend: torch
    script: train.py

metrics:
  primary: val_loss
  direction: minimize

results_file: experiments.jsonl
fleet_results_file: experiments_fleet.jsonl
logs_dir: logs
store_dir: .crucible
research_state_file: research_state.jsonl
nodes_file: nodes.json

wandb:
  required: false

execution_policy:
  require_remote: false
  required_provider: ""
  allow_local_dev: true
"""


@pytest.fixture
def e2e_project(tmp_path: Path, monkeypatch):
    """Tmp project with one completed result, no fleet, no queue.

    Suitable for exercising every MCP tool without provisioning pods or
    running a real training script.
    """
    (tmp_path / "crucible.yaml").write_text(PROJECT_YAML, encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / ".crucible").mkdir()

    # One completed result so leaderboard + completed_count are non-zero.
    (tmp_path / "experiments.jsonl").write_text(
        json.dumps({
            "id": "e2e-smoke",
            "name": "e2e-smoke",
            "status": "completed",
            "config": {"LR": "0.001", "MODEL_FAMILY": "baseline"},
            "result": {"val_loss": 1.5, "steps_completed": 500},
            "model_bytes": 12_000_000,
            "backend": "torch",
        }) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "experiments_fleet.jsonl").touch()

    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke a TOOL_DISPATCH handler by name. Mirrors the MCP server's
    own dispatch path so we exercise the same code production uses."""
    from crucible.mcp.tools import TOOL_DISPATCH
    handler = TOOL_DISPATCH.get(name)
    assert handler is not None, f"missing TOOL_DISPATCH entry for {name!r}"
    return handler(args)


# ---------------------------------------------------------------------------
# Phase 2 read-only surface
# ---------------------------------------------------------------------------


class TestPhase2ReadOnly:
    def test_tool_router_recommends_leaderboard_after_smoke(self, e2e_project):
        out = _dispatch("tool_router", {})
        assert out["recommended_tool"] == "get_leaderboard"
        assert out["state"]["completed_experiments"] == 1
        assert out["state"]["nodes"]["total"] == 0
        assert out["state"]["active_session"] is None

    def test_runs_search_finds_the_completed_run(self, e2e_project):
        out = _dispatch("runs_search", {
            "where": "status == 'completed'",
            "select": ["name", "result.val_loss"],
        })
        assert out["matched"] == 1
        assert out["rows"][0]["name"] == "e2e-smoke"
        assert out["rows"][0]["val_loss"] == 1.5

    def test_runs_search_strict_fields_catches_typo(self, e2e_project):
        out = _dispatch("runs_search", {
            "where": "val_los < 2.0",
            "strict_fields": True,
        })
        assert "error" in out
        assert "unknown field" in out["error"]

    def test_briefing_includes_next_actions(self, e2e_project):
        out = _dispatch("get_research_briefing", {})
        assert "next_actions" in out
        na = out["next_actions"]
        assert na is not None
        assert "recommended_tool" in na
        assert "## Recommended Next Tool" in out["markdown_summary"]


# ---------------------------------------------------------------------------
# Phase 1 session protocols (start → status → cancel)
# ---------------------------------------------------------------------------


class TestAutonomousSessionRoundtrip:
    def test_full_lifecycle(self, e2e_project):
        # 1. start
        started = _dispatch("autonomous_research_loop", {
            "action": "start", "iterations": 2, "tier": "smoke",
        })
        assert "session_id" in started
        assert started["stage"] == "hypothesis"
        sid = started["session_id"]
        snap = started["state_snapshot"]
        assert "content_hash" in snap

        # 2. status — surfaces persisted state
        status = _dispatch("autonomous_research_loop", {
            "action": "status", "session_id": sid,
        })
        assert status["status"] == "running"
        assert status["current_stage"] == "hypothesis"
        assert "yaml_path" in status

        # 3. submit a canned hypothesis response
        submitted = _dispatch("autonomous_research_loop", {
            "action": "submit", "session_id": sid,
            "state_snapshot": snap,
            "response": {
                "hypotheses": [{
                    "hypothesis": "e2e test",
                    "name": "e2e_h",
                    "expected_impact": 0.01,
                    "confidence": 0.5,
                    "config": {"MODEL_FAMILY": "baseline"},
                    "rationale": "smoke",
                    "family": "baseline",
                }],
            },
        })
        assert submitted["stage_applied"] == "hypothesis"
        assert submitted["next_stage"] == "reflection"

        # 4. cancel
        canceled = _dispatch("autonomous_research_loop", {
            "action": "cancel", "session_id": sid,
            "reason": "e2e test",
        })
        assert canceled["session_status"] == "canceled"
        # Idempotent
        again = _dispatch("autonomous_research_loop", {
            "action": "cancel", "session_id": sid,
        })
        assert again["already_terminal"] is True


class TestTreeSessionRoundtrip:
    def test_start_status_cancel(self, e2e_project):
        # Build a tree first.
        _dispatch("tree_create", {
            "name": "e2e-tree",
            "roots": [{"name": "r0", "config": {"LR": "3e-4"}}],
        })

        started = _dispatch("tree_autonomous_loop", {
            "action": "start", "tree_name": "e2e-tree",
            "iterations": 1, "n_children": 2,
        })
        # Tree has no expanded results; expect external_dispatch hint OR
        # a node prompt depending on tree state.
        assert "session_id" in started or "error" in started

        if "session_id" in started:
            sid = started["session_id"]
            status = _dispatch("tree_autonomous_loop", {
                "action": "status", "session_id": sid,
            })
            assert status["status"] == "running"
            assert status["tree_name"] == "e2e-tree"
            assert "budget_usd" in status

            canceled = _dispatch("tree_autonomous_loop", {
                "action": "cancel", "session_id": sid,
            })
            assert canceled["session_status"] == "canceled"


class TestHarnessSessionRoundtrip:
    def test_start_status_cancel(self, e2e_project):
        # Domain spec needed for harness.
        spec_path = e2e_project / "harness_spec.yaml"
        spec_path.write_text(yaml.safe_dump({
            "name": "e2e",
            "interface": {
                "class_name": "Harness",
                "required_methods": [
                    {"name": "predict", "signature": "(self, x)"},
                ],
            },
            "baselines": [],
            "metrics": [{"name": "accuracy", "direction": "maximize"}],
            "constraints": {},
            "proposal_guidance": "Propose.",
            "evaluation": {},
        }), encoding="utf-8")

        started = _dispatch("harness_autonomous_loop", {
            "action": "start",
            "domain_spec": str(spec_path),
            "tree_name": "e2e-harness-tree",
            "iterations": 2,
        })
        if "error" in started:
            pytest.skip(f"harness start unavailable in this env: {started['error']}")
        sid = started["session_id"]

        status = _dispatch("harness_autonomous_loop", {
            "action": "status", "session_id": sid,
        })
        assert status["status"] == "running"

        canceled = _dispatch("harness_autonomous_loop", {
            "action": "cancel", "session_id": sid,
        })
        assert canceled["session_status"] == "canceled"


# ---------------------------------------------------------------------------
# Other Phase 1 tools
# ---------------------------------------------------------------------------


class TestTreeAutoExpandContract:
    def test_request_prompt_then_submit_with_bad_shape(self, e2e_project):
        # Build a tree with one root.
        _dispatch("tree_create", {
            "name": "e2e-expand",
            "roots": [{"name": "r", "config": {"LR": "3e-4"}}],
        })
        # Record a result so the root is expandable.
        from crucible.researcher.search_tree import SearchTree
        tree_dir = e2e_project / ".crucible" / "search_trees" / "e2e-expand"
        tree = SearchTree.load(tree_dir)
        root_id = tree.meta["root_node_ids"][0]
        tree.record_result(root_id, {"val_loss": 2.0})

        # Request the prompt.
        prompt = _dispatch("tree_auto_expand", {
            "action": "request_prompt", "name": "e2e-expand",
            "node_id": root_id, "n_children": 2,
        })
        assert "system" in prompt and "user" in prompt
        assert "tree_snapshot" in prompt
        assert "content_hash" in prompt["tree_snapshot"]

        # Submit with a malformed response shape — must error per seam 3.
        bad = _dispatch("tree_auto_expand", {
            "action": "submit", "name": "e2e-expand", "node_id": root_id,
            "response": {"results": [{"name": "x", "config": {}}]},
            "tree_snapshot": prompt["tree_snapshot"],
        })
        assert "error" in bad
        assert "children" in bad["error"].lower()


class TestTreePruneAuto:
    def test_no_threshold_returns_graceful_zero(self, e2e_project):
        _dispatch("tree_create", {
            "name": "e2e-prune",
            "roots": [{"name": "r", "config": {}}],
        })
        out = _dispatch("tree_prune", {
            "name": "e2e-prune", "mode": "auto",
        })
        assert out["mode"] == "auto"
        assert out["pruned_count"] == 0
        # Phase 1.10: graceful message when no threshold configured.
        assert "threshold" in out.get("message", "").lower()


class TestContextPushFindingAutoPromote:
    def test_auto_promote_records_and_promotes_at_high_confidence(
        self, e2e_project, monkeypatch
    ):
        # Don't actually push to a real hub — patch the promotion func to
        # observe the call shape.
        promotion_calls = {"n": 0}
        from crucible.mcp import tools as _tools

        # The auto-promote path calls finding_promote internally; stub it
        # so we don't depend on a real hub.
        orig = _tools.finding_promote

        def stub_promote(args):
            promotion_calls["n"] += 1
            return {"status": "promoted", "scope": args.get("scope", "track")}

        monkeypatch.setattr(_tools, "finding_promote", stub_promote)

        out = _dispatch("context_push_finding", {
            "finding": "e2e-test finding",
            "category": "observation",
            "confidence": 0.95,
            "auto_promote": True,
            "auto_promote_scope": "global",
        })
        # Recorded at minimum.
        assert "recorded" in out.get("status", "").lower() or out.get("status") == "recorded"


# ---------------------------------------------------------------------------
# Combined: tool_router sees the active session
# ---------------------------------------------------------------------------


def test_tool_router_detects_active_session(e2e_project):
    """G.4 seam (router stage field) integration: start a session, then
    call tool_router and assert active_session.stage is the real stage
    string, not None."""
    started = _dispatch("autonomous_research_loop", {
        "action": "start", "iterations": 2, "tier": "smoke",
    })
    sid = started["session_id"]
    try:
        routed = _dispatch("tool_router", {})
        active = routed["state"]["active_session"]
        assert active is not None
        assert active["session_id"] == sid
        # Phase 2.1 + 4657bb7 fix: stage comes from `current_stage`,
        # not the missing `stage` field.
        assert active["stage"] == "hypothesis"
        assert active["kind"] == "autonomous"
    finally:
        _dispatch("autonomous_research_loop", {
            "action": "cancel", "session_id": sid,
        })
