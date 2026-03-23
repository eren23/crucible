"""Tests for MCP tree_* tool handlers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared FakeConfig
# ---------------------------------------------------------------------------


class _FakeMeta:
    primary = "val_loss"


class _FakeConfig:
    """Minimal stand-in for ProjectConfig used by tree tool handlers."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.store_dir = ".crucible"
        self.metrics = _FakeMeta()
        self.nodes_file = "nodes.json"


def _patch_config(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> None:
    """Replace ``_get_config`` in the tools module with a fake."""
    monkeypatch.setattr(
        "crucible.mcp.tools._get_config",
        lambda: _FakeConfig(project_root),
    )


# ---------------------------------------------------------------------------
# tree_create
# ---------------------------------------------------------------------------


class TestTreeCreate:
    def test_creates_empty_tree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create

        _patch_config(monkeypatch, tmp_path)

        result = tree_create({"name": "my-tree"})
        assert result["status"] == "created"
        assert result["name"] == "my-tree"
        assert result["root_node_ids"] == []
        assert result["total_nodes"] == 0

    def test_creates_tree_with_roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create

        _patch_config(monkeypatch, tmp_path)

        roots = [
            {"name": "baseline", "config": {"LR": "3e-4"}, "hypothesis": "baseline run"},
            {"name": "bigger", "config": {"MODEL_DIM": "256"}, "hypothesis": "scale up"},
        ]
        result = tree_create({"name": "rooted-tree", "roots": roots})
        assert result["status"] == "created"
        assert result["name"] == "rooted-tree"
        assert len(result["root_node_ids"]) == 2
        assert result["total_nodes"] == 2

    def test_duplicate_tree_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create

        _patch_config(monkeypatch, tmp_path)

        tree_create({"name": "dup-tree"})
        result = tree_create({"name": "dup-tree"})
        assert "error" in result
        assert "already exists" in result["error"]


# ---------------------------------------------------------------------------
# tree_get
# ---------------------------------------------------------------------------


class TestTreeGet:
    def test_returns_summary_and_ascii(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_get

        _patch_config(monkeypatch, tmp_path)

        tree_create({
            "name": "vis-tree",
            "roots": [{"name": "root", "config": {}}],
        })
        result = tree_get({"name": "vis-tree"})
        assert "summary" in result
        assert "ascii_tree" in result
        assert result["summary"]["name"] == "vis-tree"
        assert result["summary"]["total_nodes"] == 1
        assert "root" in result["ascii_tree"]

    def test_nonexistent_tree_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_get

        _patch_config(monkeypatch, tmp_path)

        result = tree_get({"name": "no-such-tree"})
        assert "error" in result


# ---------------------------------------------------------------------------
# tree_expand_node
# ---------------------------------------------------------------------------


class TestTreeExpandNode:
    def test_adds_children_and_returns_ids(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_expand_node
        from crucible.researcher.search_tree import SearchTree

        _patch_config(monkeypatch, tmp_path)

        create_res = tree_create({
            "name": "expand-tree",
            "roots": [{"name": "root", "config": {"LR": "3e-4"}}],
        })
        root_id = create_res["root_node_ids"][0]

        # Record a result on the root so it can be expanded
        tree_dir = tmp_path / ".crucible" / "search_trees" / "expand-tree"
        tree = SearchTree.load(tree_dir)
        tree.record_result(root_id, {"val_loss": 2.0})

        children = [
            {"name": "child-a", "config": {"LR": "1e-4"}, "hypothesis": "lower lr"},
            {"name": "child-b", "config": {"LR": "1e-3"}, "hypothesis": "higher lr"},
        ]
        result = tree_expand_node({
            "name": "expand-tree",
            "parent_node_id": root_id,
            "children": children,
        })
        assert result["status"] == "expanded"
        assert result["parent_node_id"] == root_id
        assert len(result["new_node_ids"]) == 2
        assert result["total_nodes"] == 3

    def test_expand_nonexistent_node_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_expand_node

        _patch_config(monkeypatch, tmp_path)
        tree_create({"name": "err-tree"})

        result = tree_expand_node({
            "name": "err-tree",
            "parent_node_id": "nonexistent",
            "children": [{"name": "c", "config": {}}],
        })
        assert "error" in result


# ---------------------------------------------------------------------------
# tree_prune
# ---------------------------------------------------------------------------


class TestTreePrune:
    def test_prune_single_node(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_prune

        _patch_config(monkeypatch, tmp_path)

        create_res = tree_create({
            "name": "prune-tree",
            "roots": [{"name": "root", "config": {}}],
        })
        root_id = create_res["root_node_ids"][0]

        result = tree_prune({
            "name": "prune-tree",
            "node_id": root_id,
            "reason": "underperforming",
        })
        assert result["status"] == "node_pruned"
        assert result["node_id"] == root_id
        assert result["total_pruned"] == 1

    def test_prune_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_prune
        from crucible.researcher.search_tree import SearchTree

        _patch_config(monkeypatch, tmp_path)

        create_res = tree_create({
            "name": "branch-tree",
            "roots": [{"name": "root", "config": {}}],
        })
        root_id = create_res["root_node_ids"][0]

        # Expand root so there are children to prune
        tree_dir = tmp_path / ".crucible" / "search_trees" / "branch-tree"
        tree = SearchTree.load(tree_dir)
        tree.record_result(root_id, {"val_loss": 2.0})
        tree.expand_node(root_id, [{"name": "child", "config": {}}])

        result = tree_prune({
            "name": "branch-tree",
            "node_id": root_id,
            "reason": "bad direction",
            "prune_branch": True,
        })
        assert result["status"] == "branch_pruned"
        assert result["nodes_pruned"] == 2


# ---------------------------------------------------------------------------
# tree_list
# ---------------------------------------------------------------------------


class TestTreeList:
    def test_lists_trees(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_list

        _patch_config(monkeypatch, tmp_path)

        tree_create({"name": "tree-alpha"})
        tree_create({"name": "tree-beta"})

        result = tree_list({})
        assert result["total"] == 2
        names = [t["name"] for t in result["trees"]]
        assert "tree-alpha" in names
        assert "tree-beta" in names

    def test_empty_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_list

        _patch_config(monkeypatch, tmp_path)

        result = tree_list({})
        assert result["trees"] == []
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# tree_enqueue_pending
# ---------------------------------------------------------------------------


class TestTreeEnqueuePending:
    def test_enqueues_pending_nodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_enqueue_pending

        _patch_config(monkeypatch, tmp_path)

        create_res = tree_create({
            "name": "enq-tree",
            "roots": [
                {"name": "exp-1", "config": {"LR": "1e-4"}},
                {"name": "exp-2", "config": {"LR": "3e-4"}},
            ],
        })

        # Mock enqueue_experiments to avoid filesystem queue side effects
        # but still return properly shaped items
        call_log: list[Any] = []

        def fake_enqueue(queue_path, experiments, limit=0):
            call_log.append(experiments)
            return [
                {"experiment_name": e["name"], "run_id": f"run_{e['name']}"}
                for e in experiments
            ]

        # Patch at the source module -- tree_enqueue_pending imports it locally
        monkeypatch.setattr(
            "crucible.fleet.queue.enqueue_experiments",
            fake_enqueue,
        )

        result = tree_enqueue_pending({"name": "enq-tree"})
        assert result["status"] == "enqueued"
        assert result["enqueued"] == 2
        assert len(result["items"]) == 2

    def test_no_pending_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_prune, tree_enqueue_pending

        _patch_config(monkeypatch, tmp_path)

        create_res = tree_create({
            "name": "nopend-tree",
            "roots": [{"name": "r", "config": {}}],
        })
        # Prune the only node
        tree_prune({
            "name": "nopend-tree",
            "node_id": create_res["root_node_ids"][0],
            "reason": "test",
        })

        result = tree_enqueue_pending({"name": "nopend-tree"})
        assert result["status"] == "no_pending_nodes"
        assert result["enqueued"] == 0


# ---------------------------------------------------------------------------
# tree_sync_results
# ---------------------------------------------------------------------------


class TestTreeSyncResults:
    def test_syncs_completed_results(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_sync_results
        from crucible.researcher.search_tree import SearchTree

        _patch_config(monkeypatch, tmp_path)

        create_res = tree_create({
            "name": "sync-tree",
            "roots": [{"name": "exp-sync", "config": {}}],
        })
        root_id = create_res["root_node_ids"][0]

        # Manually set node to queued with a run_id
        tree_dir = tmp_path / ".crucible" / "search_trees" / "sync-tree"
        tree = SearchTree.load(tree_dir)
        tree.nodes[root_id]["status"] = "queued"
        tree.nodes[root_id]["run_id"] = "run_sync_123"
        tree._save_snapshot()

        # Mock merged_results to return a completed result matching the run_id
        monkeypatch.setattr(
            "crucible.analysis.results.merged_results",
            lambda config: [
                {
                    "id": "run_sync_123",
                    "run_id": "run_sync_123",
                    "status": "completed",
                    "result": {"val_loss": 1.5},
                },
            ],
        )

        result = tree_sync_results({"name": "sync-tree"})
        assert result["status"] == "synced"
        assert result["synced_count"] == 1
        assert len(result["synced_nodes"]) == 1
        assert result["synced_nodes"][0]["run_id"] == "run_sync_123"

    def test_no_matching_results(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_sync_results

        _patch_config(monkeypatch, tmp_path)

        tree_create({
            "name": "nosync-tree",
            "roots": [{"name": "exp-ns", "config": {}}],
        })

        monkeypatch.setattr(
            "crucible.analysis.results.merged_results",
            lambda config: [],
        )

        result = tree_sync_results({"name": "nosync-tree"})
        assert result["status"] == "synced"
        assert result["synced_count"] == 0
