"""Tests for the SearchTree class."""
from __future__ import annotations

import pytest

from crucible.core.errors import SearchTreeError
from crucible.researcher.search_tree import SearchTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(tmp_path, name="test-tree", **kwargs):
    """Create a minimal tree with default settings."""
    tree_dir = tmp_path / "search_trees" / name
    return SearchTree.create(
        tree_dir=tree_dir,
        name=name,
        description="test tree",
        primary_metric="val_bpb",
        metric_direction="minimize",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Creation and roots
# ---------------------------------------------------------------------------


class TestCreateAndRoots:
    def test_create_empty_tree(self, tmp_path):
        tree = _make_tree(tmp_path)
        assert tree.meta["name"] == "test-tree"
        assert tree.meta["total_nodes"] == 0
        assert tree.meta["status"] == "active"
        assert tree.meta["root_node_ids"] == []

    def test_create_with_roots(self, tmp_path):
        tree = _make_tree(
            tmp_path,
            roots=[
                {"name": "baseline", "config": {"MODEL_DIM": "128"}, "hypothesis": "baseline run"},
                {"name": "bigger", "config": {"MODEL_DIM": "256"}, "hypothesis": "bigger model"},
            ],
        )
        assert tree.meta["total_nodes"] == 2
        assert len(tree.meta["root_node_ids"]) == 2

        for rid in tree.meta["root_node_ids"]:
            node = tree.get_node(rid)
            assert node is not None
            assert node["depth"] == 0
            assert node["parent_node_id"] is None
            assert node["status"] == "pending"

    def test_create_duplicate_raises(self, tmp_path):
        _make_tree(tmp_path)
        with pytest.raises(SearchTreeError, match="already exists"):
            _make_tree(tmp_path)

    def test_add_root(self, tmp_path):
        tree = _make_tree(tmp_path)
        node_id = tree.add_root(
            name="root-1",
            config={"MODEL_DIM": "128"},
            hypothesis="test hypothesis",
            rationale="test rationale",
            tags=["baseline"],
        )
        assert node_id in tree.nodes
        node = tree.get_node(node_id)
        assert node["experiment_name"] == "root-1"
        assert node["config"] == {"MODEL_DIM": "128"}
        assert node["hypothesis"] == "test hypothesis"
        assert node["tags"] == ["baseline"]
        assert node["depth"] == 0

    def test_add_root_max_nodes(self, tmp_path):
        tree = _make_tree(tmp_path, max_nodes=1)
        tree.add_root(name="r1", config={})
        with pytest.raises(SearchTreeError, match="max_nodes"):
            tree.add_root(name="r2", config={})


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


class TestExpansion:
    def test_expand_node(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={"LR": "3e-4", "MODEL_DIM": "128"})

        child_ids = tree.expand_node(
            root_id,
            [
                {"name": "child-a", "config": {"LR": "1e-4"}, "hypothesis": "lower lr"},
                {"name": "child-b", "config": {"LR": "1e-3"}, "hypothesis": "higher lr"},
            ],
        )

        assert len(child_ids) == 2
        assert tree.meta["total_nodes"] == 3

        for cid in child_ids:
            child = tree.get_node(cid)
            assert child["parent_node_id"] == root_id
            assert child["depth"] == 1
            assert child["config"]["MODEL_DIM"] == "128"  # inherited
            assert child["status"] == "pending"

        # Check parent's children list
        root = tree.get_node(root_id)
        assert set(root["children"]) == set(child_ids)

    def test_expand_merges_config(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={"A": "1", "B": "2"})
        child_ids = tree.expand_node(root_id, [{"name": "c", "config": {"B": "99", "C": "3"}}])
        child = tree.get_node(child_ids[0])
        assert child["config"] == {"A": "1", "B": "99", "C": "3"}

    def test_expand_sets_parent_run_id(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={})
        # Simulate a completed root with run_id
        tree.nodes[root_id]["run_id"] = "run_abc123"
        tree.nodes[root_id]["status"] = "completed"

        child_ids = tree.expand_node(root_id, [{"name": "c", "config": {}}])
        child = tree.get_node(child_ids[0])
        assert child["config"]["PARENT_RUN_ID"] == "run_abc123"

    def test_expand_pruned_node_raises(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={})
        tree.prune_node(root_id, "test")
        with pytest.raises(SearchTreeError, match="pruned"):
            tree.expand_node(root_id, [{"name": "c", "config": {}}])

    def test_expand_exceeds_max_children(self, tmp_path):
        tree = _make_tree(tmp_path, max_expansions_per_node=2)
        root_id = tree.add_root(name="root", config={})
        tree.expand_node(root_id, [{"name": "c1", "config": {}}, {"name": "c2", "config": {}}])
        with pytest.raises(SearchTreeError, match="max_expansions_per_node"):
            tree.expand_node(root_id, [{"name": "c3", "config": {}}])

    def test_expand_exceeds_max_depth(self, tmp_path):
        tree = _make_tree(tmp_path, max_depth=1)
        root_id = tree.add_root(name="root", config={})
        child_ids = tree.expand_node(root_id, [{"name": "c", "config": {}}])
        with pytest.raises(SearchTreeError, match="max_depth"):
            tree.expand_node(child_ids[0], [{"name": "gc", "config": {}}])

    def test_expand_nonexistent_parent_raises(self, tmp_path):
        tree = _make_tree(tmp_path)
        with pytest.raises(SearchTreeError, match="not found"):
            tree.expand_node("nonexistent", [{"name": "c", "config": {}}])


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------


class TestResultRecording:
    def test_record_result(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={})
        tree.record_result(root_id, {"val_bpb": 1.5, "run_id": "run_123"})

        node = tree.get_node(root_id)
        assert node["status"] == "completed"
        assert node["result_metric"] == 1.5
        assert node["run_id"] == "run_123"
        assert node["completed_at"] is not None
        assert tree.meta["completed_nodes"] == 1
        assert tree.meta["best_node_id"] == root_id
        assert tree.meta["best_metric"] == 1.5

    def test_best_tracking_minimize(self, tmp_path):
        tree = _make_tree(tmp_path)
        r1 = tree.add_root(name="r1", config={})
        r2 = tree.add_root(name="r2", config={})
        tree.record_result(r1, {"val_bpb": 2.0})
        tree.record_result(r2, {"val_bpb": 1.5})
        assert tree.meta["best_node_id"] == r2
        assert tree.meta["best_metric"] == 1.5

    def test_best_tracking_maximize(self, tmp_path):
        tree_dir = tmp_path / "search_trees" / "max-tree"
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name="max-tree",
            primary_metric="accuracy",
            metric_direction="maximize",
        )
        r1 = tree.add_root(name="r1", config={})
        r2 = tree.add_root(name="r2", config={})
        tree.record_result(r1, {"accuracy": 0.8})
        tree.record_result(r2, {"accuracy": 0.9})
        assert tree.meta["best_node_id"] == r2
        assert tree.meta["best_metric"] == 0.9

    def test_visit_count_propagation(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={})
        child_ids = tree.expand_node(root_id, [{"name": "c1", "config": {}}])
        tree.record_result(child_ids[0], {"val_bpb": 1.0})

        child = tree.get_node(child_ids[0])
        root = tree.get_node(root_id)
        assert child["visit_count"] == 1
        assert root["visit_count"] == 1

    def test_record_pruned_node_raises(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="r", config={})
        tree.prune_node(rid, "bad")
        with pytest.raises(SearchTreeError, match="pruned"):
            tree.record_result(rid, {"val_bpb": 1.0})


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPruning:
    def test_prune_node(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="r", config={})
        tree.prune_node(rid, "underperforming")
        node = tree.get_node(rid)
        assert node["status"] == "pruned"
        assert node["prune_reason"] == "underperforming"
        assert node["pruned_at"] is not None
        assert tree.meta["pruned_nodes"] == 1

    def test_prune_idempotent(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="r", config={})
        tree.prune_node(rid, "first")
        tree.prune_node(rid, "second")  # should not raise
        assert tree.meta["pruned_nodes"] == 1  # still 1

    def test_prune_branch(self, tmp_path):
        tree = _make_tree(tmp_path)
        root_id = tree.add_root(name="root", config={})
        c_ids = tree.expand_node(root_id, [
            {"name": "c1", "config": {}},
            {"name": "c2", "config": {}},
        ])
        gc_ids = tree.expand_node(c_ids[0], [{"name": "gc1", "config": {}}])

        count = tree.prune_branch(root_id, "prune all")
        assert count == 4  # root + c1 + c2 + gc1
        for nid in [root_id] + c_ids + gc_ids:
            assert tree.get_node(nid)["status"] == "pruned"


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


class TestSelectionPolicies:
    def test_agent_directed_returns_empty(self, tmp_path):
        tree = _make_tree(tmp_path, expansion_policy="agent_directed")
        tree.add_root(name="r", config={})
        assert tree.select_next(3) == []

    def test_greedy_selection(self, tmp_path):
        tree = _make_tree(tmp_path, expansion_policy="greedy")
        r1 = tree.add_root(name="r1", config={})
        r2 = tree.add_root(name="r2", config={})
        # Record results to set parent metrics, then expand
        tree.record_result(r1, {"val_bpb": 2.0})
        tree.record_result(r2, {"val_bpb": 1.0})

        # Now expand both to create pending children
        c1_ids = tree.expand_node(r1, [{"name": "c1", "config": {}}])
        c2_ids = tree.expand_node(r2, [{"name": "c2", "config": {}}])

        selected = tree.select_next(1)
        # Should pick the child of r2 (better parent metric, lower val_bpb for minimize)
        assert selected == c2_ids

    def test_ucb1_selection_unvisited_first(self, tmp_path):
        tree = _make_tree(tmp_path, expansion_policy="ucb1")
        r1 = tree.add_root(name="r1", config={})
        r2 = tree.add_root(name="r2", config={})
        # Both unvisited, should still return results
        selected = tree.select_next(2)
        assert len(selected) == 2

    def test_epsilon_greedy_selection(self, tmp_path):
        tree = _make_tree(
            tmp_path,
            expansion_policy="epsilon_greedy",
            expansion_config={"epsilon": 0.0},  # always greedy
        )
        r1 = tree.add_root(name="r1", config={})
        r2 = tree.add_root(name="r2", config={})
        tree.record_result(r1, {"val_bpb": 2.0})
        tree.record_result(r2, {"val_bpb": 1.0})

        c1_ids = tree.expand_node(r1, [{"name": "c1", "config": {}}])
        c2_ids = tree.expand_node(r2, [{"name": "c2", "config": {}}])

        selected = tree.select_next(1)
        # With epsilon=0, should be deterministically greedy (pick child of best parent)
        assert selected == c2_ids

    def test_select_next_no_pending(self, tmp_path):
        tree = _make_tree(tmp_path, expansion_policy="greedy")
        rid = tree.add_root(name="r", config={})
        tree.record_result(rid, {"val_bpb": 1.0})
        assert tree.select_next(1) == []


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_get_expandable_nodes(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="r", config={})
        assert tree.get_expandable_nodes() == []  # not completed yet

        tree.record_result(rid, {"val_bpb": 1.0})
        expandable = tree.get_expandable_nodes()
        assert len(expandable) == 1
        assert expandable[0]["node_id"] == rid

    def test_get_frontier(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="r", config={})
        frontier = tree.get_frontier()
        assert len(frontier) == 1
        assert frontier[0]["node_id"] == rid

        # After expanding, root is no longer frontier
        tree.expand_node(rid, [{"name": "c", "config": {}}])
        frontier = tree.get_frontier()
        assert len(frontier) == 1
        assert frontier[0]["node_id"] != rid

    def test_get_ancestry(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        cids = tree.expand_node(rid, [{"name": "child", "config": {}}])
        gcids = tree.expand_node(cids[0], [{"name": "grandchild", "config": {}}])

        ancestry = tree.get_ancestry(gcids[0])
        assert len(ancestry) == 3
        assert ancestry[0]["node_id"] == rid
        assert ancestry[1]["node_id"] == cids[0]
        assert ancestry[2]["node_id"] == gcids[0]

    def test_get_best_path(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        tree.record_result(rid, {"val_bpb": 2.0})
        cids = tree.expand_node(rid, [{"name": "child", "config": {}}])
        tree.record_result(cids[0], {"val_bpb": 1.5})

        best_path = tree.get_best_path()
        assert len(best_path) == 2
        assert best_path[0]["node_id"] == rid
        assert best_path[1]["node_id"] == cids[0]

    def test_get_best_path_empty(self, tmp_path):
        tree = _make_tree(tmp_path)
        assert tree.get_best_path() == []

    def test_get_siblings(self, tmp_path):
        tree = _make_tree(tmp_path)
        r1 = tree.add_root(name="r1", config={})
        r2 = tree.add_root(name="r2", config={})

        siblings = tree.get_siblings(r1)
        assert len(siblings) == 1
        assert siblings[0]["node_id"] == r2

    def test_get_siblings_children(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        cids = tree.expand_node(rid, [
            {"name": "c1", "config": {}},
            {"name": "c2", "config": {}},
            {"name": "c3", "config": {}},
        ])
        siblings = tree.get_siblings(cids[1])
        sibling_ids = {s["node_id"] for s in siblings}
        assert sibling_ids == {cids[0], cids[2]}

    def test_get_tree_summary(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="r", config={})
        tree.record_result(rid, {"val_bpb": 1.0})

        summary = tree.get_tree_summary()
        assert summary["name"] == "test-tree"
        assert summary["total_nodes"] == 1
        assert summary["completed_nodes"] == 1
        assert summary["status_breakdown"]["completed"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        tree = _make_tree(tmp_path)
        r1 = tree.add_root(name="root", config={"LR": "3e-4"}, hypothesis="test")
        c_ids = tree.expand_node(r1, [{"name": "child", "config": {"LR": "1e-4"}}])
        tree.record_result(r1, {"val_bpb": 2.0})

        # Reload from disk
        tree2 = SearchTree.load(tree.tree_dir)
        assert tree2.meta["name"] == "test-tree"
        assert tree2.meta["total_nodes"] == 2
        assert tree2.meta["completed_nodes"] == 1

        root = tree2.get_node(r1)
        assert root is not None
        assert root["experiment_name"] == "root"
        assert root["config"]["LR"] == "3e-4"
        assert root["status"] == "completed"

        child = tree2.get_node(c_ids[0])
        assert child is not None
        assert child["config"]["LR"] == "1e-4"
        assert child["parent_node_id"] == r1

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(SearchTreeError, match="No search tree"):
            SearchTree.load(tmp_path / "nonexistent")

    def test_ledger_files_created(self, tmp_path):
        tree = _make_tree(tmp_path)
        tree.add_root(name="r", config={})
        assert tree._meta_path.exists()
        assert tree._nodes_path.exists()
        assert tree._snapshot_path.exists()


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


class TestRenderAscii:
    def test_empty_tree(self, tmp_path):
        tree = _make_tree(tmp_path)
        assert tree.render_ascii() == "(empty tree)"

    def test_single_root(self, tmp_path):
        tree = _make_tree(tmp_path)
        tree.add_root(name="baseline", config={})
        output = tree.render_ascii()
        assert "baseline" in output
        assert "○" in output  # pending icon

    def test_tree_with_results(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        tree.record_result(rid, {"val_bpb": 1.5})
        cids = tree.expand_node(rid, [
            {"name": "child-a", "config": {}},
            {"name": "child-b", "config": {}},
        ])
        tree.record_result(cids[0], {"val_bpb": 1.3})

        output = tree.render_ascii()
        assert "root" in output
        assert "child-a" in output
        assert "child-b" in output
        assert "1.5000" in output
        assert "1.3000" in output

    def test_max_depth_rendering(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        cids = tree.expand_node(rid, [{"name": "child", "config": {}}])
        tree.expand_node(cids[0], [{"name": "grandchild", "config": {}}])

        output = tree.render_ascii(max_depth=1)
        assert "root" in output
        assert "child" in output
        assert "grandchild" not in output
