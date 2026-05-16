"""Tests for the tree_autonomous_loop session driver (Phase 1.4b)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError, StaleSubmitError
from crucible.researcher import tree_autonomous_session as tas
from crucible.researcher.tree_autonomous_session import (
    TreeAutonomousSession,
    TreeAutonomousSessionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_config(project_dir: Path):
    """Reuse the conftest project_dir + program.md setup."""
    os.chdir(project_dir)
    (project_dir / "program.md").write_text("Minimise val_loss.", encoding="utf-8")
    return load_config()


def _make_expandable_tree(config) -> str:
    """Create a tree with a root that's been completed and is expandable."""
    from crucible.researcher.search_tree import SearchTree
    tree_dir = config.project_root / ".crucible" / "search_trees" / "test-tree"
    tree = SearchTree.create(
        tree_dir=tree_dir,
        name="test-tree",
        description="test",
        primary_metric="val_loss",
        metric_direction="minimize",
    )
    root_id = tree.add_root(name="root", config={"LR": "3e-4"})
    tree.record_result(root_id, {"val_loss": 2.0})
    return "test-tree"


def _canned_expansion_response() -> list[dict[str, Any]]:
    return [
        {"name": "low_lr", "config": {"LR": "1e-4"}, "hypothesis": "lower"},
        {"name": "high_lr", "config": {"LR": "1e-3"}, "hypothesis": "higher"},
    ]


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


class TestStart:
    def test_creates_session_and_returns_first_prompt(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        out = tas.action_start(project_config, tree_name=tree_name, iterations=3)
        assert "session_id" in out
        assert out["iteration"] == 0
        assert "node_id" in out
        assert isinstance(out["system"], str) and len(out["system"]) > 0
        assert isinstance(out["user"], str)
        assert out["schema"]["type"] == "array"
        assert "tree_snapshot" in out
        assert out["session_status"] == TreeAutonomousSession.STATUS_RUNNING
        assert out["iterations_planned"] == 3

    def test_idempotent_second_start_returns_active_session(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=3)
        second = tas.action_start(project_config, tree_name=tree_name, iterations=3)
        assert first["session_id"] == second["session_id"]

    def test_rejects_zero_iterations(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        with pytest.raises(CrucibleError, match="iterations must be >= 1"):
            tas.action_start(project_config, tree_name=tree_name, iterations=0)

    def test_persists_session_file(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        out = tas.action_start(project_config, tree_name=tree_name, iterations=2)
        sid = out["session_id"]
        path = (
            project_config.project_root
            / ".crucible" / "tree_autonomous_sessions" / f"{sid}.yaml"
        )
        assert path.exists()


# ---------------------------------------------------------------------------
# submit happy path
# ---------------------------------------------------------------------------


class TestSubmitFlow:
    def test_submit_expands_and_advances(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=3)
        sid = first["session_id"]
        result = tas.action_submit(
            project_config,
            session_id=sid,
            response=_canned_expansion_response(),
            tree_snapshot=first["tree_snapshot"],
        )
        assert result["session_status"] == TreeAutonomousSession.STATUS_RUNNING
        assert result["node_id"] == first["node_id"]
        assert len(result["new_node_ids"]) == 2

    def test_session_done_when_iterations_complete(self, project_config):
        """After N submits, session reaches DONE — even if more expandable
        nodes exist."""
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=1)
        sid = first["session_id"]
        result = tas.action_submit(
            project_config,
            session_id=sid,
            response=_canned_expansion_response(),
            tree_snapshot=first["tree_snapshot"],
        )
        assert result["session_status"] == TreeAutonomousSession.STATUS_DONE
        # Single-iteration sessions don't return a next prompt (the response
        # is a terminal apply, not a continuation).
        assert result.get("next_prompt") is None or result.get("next_prompt") == {}

    def test_submit_falls_back_to_session_snapshot(self, project_config):
        """If caller omits tree_snapshot, session uses last_tree_snapshot."""
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=2)
        sid = first["session_id"]
        # Submit without tree_snapshot — should auto-load from session.
        result = tas.action_submit(
            project_config,
            session_id=sid,
            response=_canned_expansion_response(),
        )
        assert result["session_status"] == TreeAutonomousSession.STATUS_RUNNING


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


class TestSubmitErrors:
    def test_unknown_session_raises(self, project_config):
        with pytest.raises(TreeAutonomousSessionError, match="not found"):
            tas.action_status(project_config, session_id="not-a-real-uuid")

    def test_stale_tree_snapshot_raises(self, project_config):
        """If a peer expands the tree between prompt and submit, the snapshot
        check trips."""
        from crucible.researcher.search_tree import SearchTree

        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=3)
        sid = first["session_id"]
        stale_snapshot = first["tree_snapshot"]

        # Peer process: load tree, expand the same node (persists on disk).
        tree_dir = project_config.project_root / ".crucible" / "search_trees" / tree_name
        peer = SearchTree.load(tree_dir)
        peer.expand_node(first["node_id"], [{"name": "peer", "config": {}, "hypothesis": ""}])

        with pytest.raises(StaleSubmitError, match="advanced"):
            tas.action_submit(
                project_config,
                session_id=sid,
                response=_canned_expansion_response(),
                tree_snapshot=stale_snapshot,
            )

    def test_cannot_submit_to_canceled_session(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=2)
        sid = first["session_id"]
        tas.action_cancel(project_config, session_id=sid)
        with pytest.raises(TreeAutonomousSessionError, match="canceled"):
            tas.action_submit(
                project_config,
                session_id=sid,
                response=_canned_expansion_response(),
            )


# ---------------------------------------------------------------------------
# status + cancel
# ---------------------------------------------------------------------------


class TestStatusAndCancel:
    def test_status_returns_state(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=3)
        sid = first["session_id"]
        status = tas.action_status(project_config, session_id=sid)
        assert status["session_id"] == sid
        assert status["tree_name"] == tree_name
        assert status["status"] == TreeAutonomousSession.STATUS_RUNNING

    def test_cancel_marks_canceled(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=2)
        sid = first["session_id"]
        out = tas.action_cancel(project_config, session_id=sid, reason="testing")
        assert out["session_status"] == TreeAutonomousSession.STATUS_CANCELED
        assert out["already_terminal"] is False
        # Idempotent
        again = tas.action_cancel(project_config, session_id=sid)
        assert again["already_terminal"] is True
