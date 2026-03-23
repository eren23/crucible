"""Tests for the TreeSearchResearcher and tree-aware hypothesis generation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from crucible.core.errors import SearchTreeError
from crucible.researcher.hypothesis import (
    _build_tree_context,
    _parse_tree_children,
    generate_tree_hypotheses,
)
from crucible.researcher.search_tree import SearchTree
from crucible.researcher.tree_loop import TreeSearchResearcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_config(tmp_path: Path) -> Any:
    """Build a minimal ProjectConfig-like object for testing."""
    cfg = SimpleNamespace(
        name="test-project",
        project_root=tmp_path,
        metrics=SimpleNamespace(primary="val_bpb", secondary=None),
        researcher=SimpleNamespace(
            model="test-model",
            budget_hours=10.0,
            max_iterations=5,
            program_file="program.md",
        ),
    )
    return cfg


def _make_tree(tmp_path: Path, name: str = "test-tree", **kwargs) -> SearchTree:
    """Create a minimal search tree."""
    tree_dir = tmp_path / "search_trees" / name
    return SearchTree.create(
        tree_dir=tree_dir,
        name=name,
        description="test tree",
        primary_metric="val_bpb",
        metric_direction="minimize",
        **kwargs,
    )


def _make_mock_llm(response: dict[str, Any]) -> MagicMock:
    """Create a mock LLM that returns a JSON response."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps(response)
    return llm


# ---------------------------------------------------------------------------
# TreeSearchResearcher: creation
# ---------------------------------------------------------------------------


class TestTreeSearchResearcherCreation:
    def test_create_new_tree(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="new-tree",
            llm=MagicMock(),
        )
        assert researcher.tree.meta["name"] == "new-tree"
        assert researcher.tree.meta["primary_metric"] == "val_bpb"

    def test_load_existing_tree(self, tmp_path):
        config = _fake_config(tmp_path)
        tree_dir = tmp_path / ".crucible" / "search_trees" / "existing-tree"

        # Create a tree first
        tree = SearchTree.create(
            tree_dir=tree_dir,
            name="existing-tree",
            description="pre-existing",
            primary_metric="val_bpb",
            metric_direction="minimize",
        )
        tree.add_root(name="root", config={"LR": "3e-4"})

        # Load it via TreeSearchResearcher
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="existing-tree",
            llm=MagicMock(),
        )
        assert researcher.tree.meta["total_nodes"] == 1
        assert researcher.tree.meta["name"] == "existing-tree"

    def test_custom_tree_dir(self, tmp_path):
        config = _fake_config(tmp_path)
        custom_dir = tmp_path / "custom_trees" / "my-tree"
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="my-tree",
            tree_dir=custom_dir,
            llm=MagicMock(),
        )
        assert researcher.tree_dir == custom_dir
        assert researcher.tree.meta["name"] == "my-tree"


# ---------------------------------------------------------------------------
# TreeSearchResearcher: run_iteration
# ---------------------------------------------------------------------------


class TestRunIteration:
    def test_no_expandable_nodes_returns_message(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="empty-tree",
            llm=MagicMock(),
        )
        summary = researcher.run_iteration()
        assert "No expandable nodes" in summary["message"]
        assert summary["expanded_nodes"] == []
        assert summary["new_children"] == []

    def test_pending_nodes_reported(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="pending-tree",
            llm=MagicMock(),
        )
        researcher.tree.add_root(name="root", config={"LR": "3e-4"})

        summary = researcher.run_iteration()
        # Root is pending, not completed, so not expandable
        assert len(summary["enqueued"]) == 1

    def test_expand_completed_node(self, tmp_path):
        config = _fake_config(tmp_path)
        llm_response = {
            "children": [
                {
                    "name": "lower_lr",
                    "config": {"LR": "1e-4"},
                    "hypothesis": "Lower LR might converge better",
                    "rationale": "Current LR may be too high",
                    "confidence": 0.7,
                },
                {
                    "name": "higher_lr",
                    "config": {"LR": "1e-3"},
                    "hypothesis": "Higher LR might find better minimum",
                    "rationale": "Explore faster learning",
                    "confidence": 0.5,
                },
            ]
        }
        mock_llm = _make_mock_llm(llm_response)

        researcher = TreeSearchResearcher(
            config=config,
            tree_name="expand-tree",
            n_children=2,
            llm=mock_llm,
        )
        root_id = researcher.tree.add_root(name="root", config={"LR": "3e-4"})
        researcher.tree.record_result(root_id, {"val_bpb": 1.5})

        summary = researcher.run_iteration()

        assert root_id in summary["expanded_nodes"]
        assert len(summary["new_children"]) == 2
        # New children should be pending
        assert len(summary["enqueued"]) == 2


# ---------------------------------------------------------------------------
# TreeSearchResearcher: sync_results
# ---------------------------------------------------------------------------


class TestSyncResults:
    def test_sync_matches_by_name(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="sync-tree",
            llm=MagicMock(),
        )
        root_id = researcher.tree.add_root(name="exp_1", config={"LR": "3e-4"})

        result = researcher.sync_results([
            {"name": "exp_1", "val_bpb": 1.3, "run_id": "run_abc"},
        ])

        assert result["synced_count"] == 1
        node = researcher.tree.get_node(root_id)
        assert node["status"] == "completed"
        assert node["result_metric"] == 1.3

    def test_sync_no_match(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="sync-tree-2",
            llm=MagicMock(),
        )
        researcher.tree.add_root(name="exp_1", config={})

        result = researcher.sync_results([
            {"name": "nonexistent", "val_bpb": 1.0},
        ])
        assert result["synced_count"] == 0


# ---------------------------------------------------------------------------
# TreeSearchResearcher: auto_prune
# ---------------------------------------------------------------------------


class TestAutoPrune:
    def test_prune_above_threshold(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="prune-tree",
            llm=MagicMock(),
        )
        r1 = researcher.tree.add_root(name="good", config={})
        r2 = researcher.tree.add_root(name="bad", config={})
        researcher.tree.record_result(r1, {"val_bpb": 1.0})
        researcher.tree.record_result(r2, {"val_bpb": 2.5})

        result = researcher.auto_prune(threshold=2.0)
        assert result["pruned_count"] == 1
        assert r2 in result["pruned_node_ids"]

    def test_no_threshold_skips(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="no-prune-tree",
            llm=MagicMock(),
        )
        researcher.tree.add_root(name="r", config={})
        result = researcher.auto_prune()
        assert result["pruned_count"] == 0


# ---------------------------------------------------------------------------
# TreeSearchResearcher: get_status / get_pending_configs
# ---------------------------------------------------------------------------


class TestStatusAndConfigs:
    def test_get_status(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="status-tree",
            max_iterations=10,
            llm=MagicMock(),
        )
        researcher.tree.add_root(name="r", config={"LR": "3e-4"})

        status = researcher.get_status()
        assert status["name"] == "status-tree"
        assert status["total_nodes"] == 1
        assert status["max_iterations"] == 10
        assert status["pending_nodes"] == 1

    def test_get_pending_configs(self, tmp_path):
        config = _fake_config(tmp_path)
        researcher = TreeSearchResearcher(
            config=config,
            tree_name="configs-tree",
            llm=MagicMock(),
        )
        researcher.tree.add_root(name="exp_a", config={"LR": "3e-4", "MODEL_DIM": "128"})

        configs = researcher.get_pending_configs()
        assert len(configs) == 1
        assert configs[0]["name"] == "exp_a"
        assert configs[0]["config"]["LR"] == "3e-4"
        assert f"tree:configs-tree" in configs[0]["tags"]


# ---------------------------------------------------------------------------
# Tree-aware context building
# ---------------------------------------------------------------------------


class TestBuildTreeContext:
    def test_context_includes_summary(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={"LR": "3e-4"})
        tree.record_result(rid, {"val_bpb": 1.5})

        ctx = _build_tree_context(tree, rid)
        assert "Tree Summary" in ctx
        assert "test-tree" in ctx
        assert "1.5" in ctx

    def test_context_includes_ancestry(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={"LR": "3e-4"})
        tree.record_result(rid, {"val_bpb": 2.0})
        cids = tree.expand_node(rid, [{"name": "child", "config": {"LR": "1e-4"}}])
        tree.record_result(cids[0], {"val_bpb": 1.5})

        ctx = _build_tree_context(tree, cids[0])
        assert "Ancestry Path" in ctx
        assert "root" in ctx
        assert "child" in ctx

    def test_context_includes_siblings(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        tree.record_result(rid, {"val_bpb": 2.0})
        cids = tree.expand_node(rid, [
            {"name": "sib_a", "config": {}, "hypothesis": "try A"},
            {"name": "sib_b", "config": {}, "hypothesis": "try B"},
        ])
        tree.record_result(cids[0], {"val_bpb": 1.8})

        ctx = _build_tree_context(tree, cids[1])
        assert "Sibling Results" in ctx
        assert "sib_a" in ctx

    def test_context_nonexistent_node(self, tmp_path):
        tree = _make_tree(tmp_path)
        ctx = _build_tree_context(tree, "nonexistent")
        assert "not found" in ctx

    def test_context_includes_best_path(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})
        tree.record_result(rid, {"val_bpb": 2.0})
        cids = tree.expand_node(rid, [{"name": "child", "config": {}}])
        tree.record_result(cids[0], {"val_bpb": 1.5})

        ctx = _build_tree_context(tree, cids[0])
        assert "Best Path" in ctx


# ---------------------------------------------------------------------------
# Tree children parsing
# ---------------------------------------------------------------------------


class TestParseTreeChildren:
    def test_parse_valid_children(self):
        text = json.dumps({
            "children": [
                {
                    "name": "child_a",
                    "config": {"LR": "1e-4"},
                    "hypothesis": "Lower LR",
                    "rationale": "May converge better",
                    "confidence": 0.8,
                },
            ]
        })
        result = _parse_tree_children(text, "parent")
        assert len(result) == 1
        assert result[0]["name"] == "child_a"
        assert result[0]["config"]["LR"] == "1e-4"
        assert result[0]["generation_method"] == "llm_tree"

    def test_parse_coerces_config_to_strings(self):
        text = json.dumps({
            "children": [
                {
                    "name": "c",
                    "config": {"LR": 0.001, "LAYERS": 12},
                    "hypothesis": "test",
                },
            ]
        })
        result = _parse_tree_children(text, "p")
        assert result[0]["config"]["LR"] == "0.001"
        assert result[0]["config"]["LAYERS"] == "12"

    def test_parse_empty_config_skipped(self):
        text = json.dumps({
            "children": [
                {"name": "c", "config": {}, "hypothesis": "no config"},
            ]
        })
        result = _parse_tree_children(text, "p")
        assert len(result) == 0

    def test_parse_fallback_hypotheses_key(self):
        text = json.dumps({
            "hypotheses": [
                {"name": "h1", "config": {"LR": "1e-4"}, "hypothesis": "test"},
            ]
        })
        result = _parse_tree_children(text, "p")
        assert len(result) == 1

    def test_parse_invalid_json(self):
        result = _parse_tree_children("not valid json at all", "p")
        assert result == []

    def test_parse_names_default(self):
        text = json.dumps({
            "children": [
                {"config": {"LR": "1e-4"}, "hypothesis": "test"},
            ]
        })
        result = _parse_tree_children(text, "parent")
        assert result[0]["name"] == "parent_child_0"


# ---------------------------------------------------------------------------
# generate_tree_hypotheses
# ---------------------------------------------------------------------------


class TestGenerateTreeHypotheses:
    def test_generates_children(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={"LR": "3e-4"})
        tree.record_result(rid, {"val_bpb": 1.5})

        llm_response = {
            "children": [
                {
                    "name": "lower_lr",
                    "config": {"LR": "1e-4"},
                    "hypothesis": "Lower LR",
                    "rationale": "May converge better",
                    "confidence": 0.7,
                },
                {
                    "name": "higher_lr",
                    "config": {"LR": "1e-3"},
                    "hypothesis": "Higher LR",
                    "rationale": "May find better minimum",
                    "confidence": 0.5,
                },
            ]
        }
        mock_llm = _make_mock_llm(llm_response)

        children = generate_tree_hypotheses(tree, rid, mock_llm, n_children=2)
        assert len(children) == 2
        assert children[0]["name"] == "lower_lr"
        assert children[1]["name"] == "higher_lr"

        # Verify LLM was called
        mock_llm.complete.assert_called_once()

    def test_returns_empty_on_llm_failure(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={})

        mock_llm = MagicMock()
        mock_llm.complete.return_value = None

        children = generate_tree_hypotheses(tree, rid, mock_llm)
        assert children == []

    def test_returns_empty_for_missing_node(self, tmp_path):
        tree = _make_tree(tmp_path)
        mock_llm = MagicMock()

        children = generate_tree_hypotheses(tree, "nonexistent", mock_llm)
        assert children == []

    def test_limits_to_n_children(self, tmp_path):
        tree = _make_tree(tmp_path)
        rid = tree.add_root(name="root", config={"LR": "3e-4"})
        tree.record_result(rid, {"val_bpb": 1.5})

        llm_response = {
            "children": [
                {"name": f"c{i}", "config": {"LR": f"{i}e-4"}, "hypothesis": f"test {i}"}
                for i in range(5)
            ]
        }
        mock_llm = _make_mock_llm(llm_response)

        children = generate_tree_hypotheses(tree, rid, mock_llm, n_children=2)
        assert len(children) == 2
