"""Tests for tree_expand_grpo MCP tool.

Orchestrator passes N pre-scored candidates (each with ``judge_score``).
Tool computes group-relative advantages, picks top-K, expands tree with
kept children, and stores ``group_advantage`` on each new node.

Judge separation is enforced when ``ProjectConfig.judges`` is configured.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.config import JudgeConfig, JudgePanel, ProjectConfig
from crucible.core.errors import ConfigError
from crucible.researcher.search_tree import SearchTree


@pytest.fixture
def project_with_tree(tmp_path: Path, monkeypatch):
    proj_root = tmp_path / "proj"
    proj_root.mkdir()
    (proj_root / ".crucible").mkdir()

    config = ProjectConfig(name="test-proj", project_root=proj_root)
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)

    tree_dir = proj_root / ".crucible" / "search_trees" / "tree1"
    tree = SearchTree.create(
        tree_dir=tree_dir,
        name="tree1",
        description="grpo test",
        primary_metric="val_bpb",
        metric_direction="minimize",
        roots=[
            {"name": "baseline", "config": {"MODEL_DIM": "128"}, "hypothesis": "baseline"},
        ],
    )
    parent_id = tree.meta["root_node_ids"][0]
    return config, tree, parent_id


class TestTreeExpandGrpo:
    def test_picks_top_k_and_stores_advantage(self, project_with_tree):
        from crucible.mcp.tools import tree_expand_grpo

        config, tree, parent_id = project_with_tree
        candidates = [
            {"name": "c1", "config": {"X": "1"}, "hypothesis": "h1", "judge_score": 0.2},
            {"name": "c2", "config": {"X": "2"}, "hypothesis": "h2", "judge_score": 0.9},
            {"name": "c3", "config": {"X": "3"}, "hypothesis": "h3", "judge_score": 0.5},
            {"name": "c4", "config": {"X": "4"}, "hypothesis": "h4", "judge_score": 0.7},
        ]

        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": parent_id,
            "candidates": candidates,
            "top_k": 2,
            "advantage_normalization": "z_score",
        })

        assert "error" not in out, out
        assert out["status"] == "expanded"
        assert len(out["new_node_ids"]) == 2

        # Reload tree from disk to verify persistence.
        tree_dir = config.project_root / ".crucible" / "search_trees" / "tree1"
        reloaded = SearchTree.load(tree_dir)
        for nid in out["new_node_ids"]:
            node = reloaded.get_node(nid)
            assert node is not None
            assert node["generation_method"] == "grpo"
            assert node.get("group_advantage") is not None
            # Highest-scoring candidates kept → c2 (0.9) and c4 (0.7).
            assert node["experiment_name"] in {"c2", "c4"}

    def test_min_max_normalization_supported(self, project_with_tree):
        from crucible.mcp.tools import tree_expand_grpo

        _, _, parent_id = project_with_tree
        candidates = [
            {"name": "c1", "config": {"X": "1"}, "judge_score": 0.0},
            {"name": "c2", "config": {"X": "2"}, "judge_score": 1.0},
            {"name": "c3", "config": {"X": "3"}, "judge_score": 0.5},
        ]
        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": parent_id,
            "candidates": candidates,
            "top_k": 2,
            "advantage_normalization": "min_max",
        })
        assert "error" not in out
        assert len(out["new_node_ids"]) == 2

    def test_rejects_candidates_without_score(self, project_with_tree):
        from crucible.mcp.tools import tree_expand_grpo

        _, _, parent_id = project_with_tree
        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": parent_id,
            "candidates": [{"name": "c1", "config": {"X": "1"}}],  # no score
            "top_k": 1,
        })
        assert "error" in out
        assert "judge_score" in out["error"]

    def test_rejects_candidates_without_name(self, project_with_tree):
        from crucible.mcp.tools import tree_expand_grpo

        _, _, parent_id = project_with_tree
        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": parent_id,
            "candidates": [{"config": {"X": "1"}, "judge_score": 0.5}],
            "top_k": 1,
        })
        assert "error" in out
        assert "name" in out["error"].lower()

    def test_rejects_unknown_parent_with_clear_message(self, project_with_tree):
        from crucible.mcp.tools import tree_expand_grpo

        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": "no-such-node",
            "candidates": [
                {"name": "c1", "config": {"X": "1"}, "judge_score": 0.5},
            ],
            "top_k": 1,
        })
        assert "error" in out
        assert "no-such-node" in out["error"]

    def test_judge_separation_enforced_when_configured(
        self, tmp_path: Path, monkeypatch,
    ):
        from crucible.mcp.tools import tree_expand_grpo

        proj_root = tmp_path / "proj"
        proj_root.mkdir()
        (proj_root / ".crucible").mkdir()

        # Mis-separated panel — same model on both judges.
        bad_panel = JudgePanel(
            reward_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        config = ProjectConfig(name="x", project_root=proj_root, judges=bad_panel)
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)

        tree_dir = proj_root / ".crucible" / "search_trees" / "tree1"
        tree = SearchTree.create(
            tree_dir=tree_dir, name="tree1",
            primary_metric="val_bpb", metric_direction="minimize",
            roots=[{"name": "r", "config": {"X": "1"}, "hypothesis": "h"}],
        )
        parent_id = tree.meta["root_node_ids"][0]

        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": parent_id,
            "candidates": [
                {"name": "c1", "config": {"X": "1"}, "judge_score": 0.5},
                {"name": "c2", "config": {"X": "2"}, "judge_score": 0.6},
            ],
            "top_k": 1,
        })
        assert "error" in out
        assert "judge" in out["error"].lower()

    def test_passes_when_judges_separated(self, tmp_path: Path, monkeypatch):
        from crucible.mcp.tools import tree_expand_grpo

        proj_root = tmp_path / "proj"
        proj_root.mkdir()
        (proj_root / ".crucible").mkdir()

        good_panel = JudgePanel(
            reward_judge=JudgeConfig(model="gemini-2.5-flash", family="gemini"),
            eval_judge=JudgeConfig(model="claude-opus-4-7", family="claude"),
        )
        config = ProjectConfig(name="x", project_root=proj_root, judges=good_panel)
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)

        tree_dir = proj_root / ".crucible" / "search_trees" / "tree1"
        tree = SearchTree.create(
            tree_dir=tree_dir, name="tree1",
            primary_metric="val_bpb", metric_direction="minimize",
            roots=[{"name": "r", "config": {"X": "1"}, "hypothesis": "h"}],
        )
        parent_id = tree.meta["root_node_ids"][0]

        out = tree_expand_grpo({
            "name": "tree1",
            "parent_node_id": parent_id,
            "candidates": [
                {"name": "c1", "config": {"X": "1"}, "judge_score": 0.5},
                {"name": "c2", "config": {"X": "2"}, "judge_score": 0.9},
            ],
            "top_k": 1,
        })
        assert "error" not in out
        assert len(out["new_node_ids"]) == 1
