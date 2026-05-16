"""Tests for the tree_autonomous_loop session driver (Phase 1.4b)."""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import load_config
from crucible.core.errors import CrucibleError, StaleSubmitError
from crucible.researcher import tree_autonomous_session as tas
from crucible.researcher.tree_autonomous_session import (
    TreeAutonomousSession,
    TreeAutonomousSessionError,
    TreeDoomLoopDetected,
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
        nodes exist. Also: iterations_completed and current_iteration
        are synced even on terminal."""
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
        # Terminal submit returns next_prompt: None (not {}).
        assert result["next_prompt"] is None

        # Codex sync fix: a 1-iteration session ends with both counters at 1,
        # not iterations_completed=1 / current_iteration=0.
        status = tas.action_status(project_config, session_id=sid)
        assert status["iterations_completed"] == 1
        assert status["current_iteration"] == 1

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

    def test_budget_exceeded_auto_cancels(self, project_config, monkeypatch):
        """Phase 1.8 propagation: tree session also enforces budget cap
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
        tree_name = _make_expandable_tree(project_config)
        with pytest.raises(BudgetExceeded):
            tas.action_start(
                project_config, tree_name=tree_name, iterations=3, budget_usd=5.0,
            )

    def test_budget_check_after_submit_under_session_lock(
        self, project_config, monkeypatch
    ):
        """Defense-in-depth: budget refresh also fires after a successful
        submit, mirroring autonomous_session.py. Even though tree-search
        submit doesn't dispatch synchronously, wall-clock cost may have
        jumped while the orchestrator deliberated."""
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
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(
            project_config, tree_name=tree_name,
            iterations=5, budget_usd=5.0,
        )
        sid = first["session_id"]
        with pytest.raises(BudgetExceeded):
            tas.action_submit(
                project_config, session_id=sid,
                response=_canned_expansion_response(),
                tree_snapshot=first["tree_snapshot"],
            )

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


# ---------------------------------------------------------------------------
# external_dispatch + continue action (Codex review fix)
# ---------------------------------------------------------------------------


class TestExternalDispatchAndContinue:
    def test_external_dispatch_hint_when_pending_exist(self, project_config):
        """When no expandable nodes but pending nodes exist, build returns
        next_action='external_dispatch' rather than marking DONE."""
        from crucible.researcher.search_tree import SearchTree

        # Build a tree with an UN-expandable pending root (no recorded result).
        tree_dir = project_config.project_root / ".crucible" / "search_trees" / "pending-tree"
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name="pending-tree",
            description="test",
            primary_metric="val_loss",
            metric_direction="minimize",
        )
        tree.add_root(name="root", config={"LR": "3e-4"})
        # No record_result — root stays pending, not expandable.

        out = tas.action_start(project_config, tree_name="pending-tree", iterations=3)
        assert out["next_action"] == "external_dispatch"
        assert "tree_enqueue_pending" in out["message"]
        # No 'action=continue' should appear in the hint as a string — it
        # was a contract bug before; we expect 'continue' instead.
        assert "submit with action=continue" not in out["message"]
        assert "action='continue'" in out["message"]

    def test_continue_action_re_runs_build_next_prompt(self, project_config):
        """After external_dispatch, calling 'continue' re-checks the tree.
        If a peer recorded a result in the meantime, the next expandable
        node's prompt is returned."""
        from crucible.researcher.search_tree import SearchTree

        tree_dir = project_config.project_root / ".crucible" / "search_trees" / "cont-tree"
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name="cont-tree",
            description="test",
            primary_metric="val_loss",
            metric_direction="minimize",
        )
        root_id = tree.add_root(name="root", config={"LR": "3e-4"})
        # Start in pending state.

        first = tas.action_start(project_config, tree_name="cont-tree", iterations=3)
        assert first["next_action"] == "external_dispatch"
        sid = first["session_id"]

        # Peer records the result — root is now expandable.
        peer = SearchTree.load(tree_dir)
        peer.record_result(root_id, {"val_loss": 2.0})

        # continue re-checks and now returns an actual prompt.
        next_prompt = tas.action_continue(project_config, session_id=sid)
        assert next_prompt["node_id"] == root_id
        assert "system" in next_prompt and "user" in next_prompt

    def test_continue_on_terminal_session_raises(self, project_config):
        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=1)
        sid = first["session_id"]
        tas.action_submit(
            project_config,
            session_id=sid,
            response=_canned_expansion_response(),
            tree_snapshot=first["tree_snapshot"],
        )
        # Session is DONE.
        with pytest.raises(TreeAutonomousSessionError, match="done|canceled|error"):
            tas.action_continue(project_config, session_id=sid)


# ---------------------------------------------------------------------------
# Doom-loop detection (Codex review gap)
# ---------------------------------------------------------------------------


class TestDoomLoop:
    def test_repeated_fingerprint_aborts(self, project_config, monkeypatch):
        """Force every prompt build to produce the same fingerprint —
        after 5 iterations the session errors out.

        Uses a 1-child response so we don't hit max_expansions_per_node (5)
        before the doom-loop window (5) trips."""
        monkeypatch.setattr(tas, "_fingerprint", lambda system, user: "STUCK")

        tree_name = _make_expandable_tree(project_config)
        first = tas.action_start(project_config, tree_name=tree_name, iterations=99)
        sid = first["session_id"]
        latest = first
        single_child = [{"name": "c", "config": {}, "hypothesis": ""}]

        with pytest.raises(TreeDoomLoopDetected):
            for _ in range(10):
                latest = tas.action_submit(
                    project_config,
                    session_id=sid,
                    response=single_child,
                    tree_snapshot=latest.get("tree_snapshot"),
                )
                if latest.get("session_status") in (
                    TreeAutonomousSession.STATUS_DONE,
                    TreeAutonomousSession.STATUS_ERROR,
                ):
                    break
                next_prompt = latest.get("next_prompt")
                if next_prompt is None:
                    break
                latest = next_prompt


# ---------------------------------------------------------------------------
# Concurrency — create-time lock (Codex review gap)
# ---------------------------------------------------------------------------


def _concurrent_tree_start_worker(project_dir_str: str, tree_name: str, queue) -> None:
    """Subprocess target: cd into project, start a tree session, put session_id."""
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(project_dir_str))
    from crucible.core.config import load_config as _lc
    from crucible.researcher import tree_autonomous_session as _tas
    config = _lc()
    out = _tas.action_start(config, tree_name=tree_name, iterations=2)
    queue.put(out["session_id"])


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fcntl locks are POSIX-only; create-lock degrades to no-op on Windows.",
)
class TestStartConcurrency:
    def test_concurrent_tree_starts_produce_single_session(self, project_dir):
        """Three processes call action_start against the same tree
        simultaneously — exactly one session is created."""
        os.chdir(project_dir)
        (project_dir / "program.md").write_text("Minimise val_loss.", encoding="utf-8")

        # Build an expandable tree in the project.
        from crucible.researcher.search_tree import SearchTree
        tree_dir = project_dir / ".crucible" / "search_trees" / "race-tree"
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name="race-tree",
            description="test",
            primary_metric="val_loss",
            metric_direction="minimize",
        )
        root_id = tree.add_root(name="root", config={"LR": "3e-4"})
        tree.record_result(root_id, {"val_loss": 2.0})

        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_concurrent_tree_start_worker,
                args=(str(project_dir), "race-tree", queue),
            )
            for _ in range(3)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=20.0)
            assert p.exitcode == 0, f"worker exitcode={p.exitcode}"

        ids: set[str] = set()
        while not queue.empty():
            ids.add(queue.get())
        assert len(ids) == 1, (
            f"expected exactly one session_id across concurrent tree starts; got {ids}"
        )
