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


class _FakeWandb:
    project = "test-project"
    entity = "test-entity"


class _FakeConfig:
    """Minimal stand-in for ProjectConfig used by tree tool handlers."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.store_dir = ".crucible"
        self.metrics = _FakeMeta()
        self.nodes_file = "nodes.json"
        self.wandb = _FakeWandb()
        # Needed by context_push_finding (Phase 1.6 auto-promote tests).
        self.research_state_file = "research_state.jsonl"


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

    def test_duplicate_tree_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.core.errors import SearchTreeError
        from crucible.mcp.tools import tree_create

        _patch_config(monkeypatch, tmp_path)

        tree_create({"name": "dup-tree"})
        with pytest.raises(SearchTreeError, match="already exists"):
            tree_create({"name": "dup-tree"})


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

    def test_nonexistent_tree_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.core.errors import SearchTreeError
        from crucible.mcp.tools import tree_get

        _patch_config(monkeypatch, tmp_path)

        with pytest.raises(SearchTreeError):
            tree_get({"name": "no-such-tree"})


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

    def test_expand_nonexistent_node_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.core.errors import SearchTreeError
        from crucible.mcp.tools import tree_create, tree_expand_node

        _patch_config(monkeypatch, tmp_path)
        tree_create({"name": "err-tree"})

        with pytest.raises(SearchTreeError):
            tree_expand_node({
                "name": "err-tree",
                "parent_node_id": "nonexistent",
                "children": [{"name": "c", "config": {}}],
            })


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

    def test_prune_auto_mode_no_threshold_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Phase 1.10: tree_prune(mode='auto') with no threshold configured
        returns pruned_count=0 with a clear message rather than erroring."""
        from crucible.mcp.tools import tree_create, tree_prune

        _patch_config(monkeypatch, tmp_path)
        tree_create({"name": "auto-noth"})

        result = tree_prune({"name": "auto-noth", "mode": "auto"})
        assert result["mode"] == "auto"
        assert result["pruned_count"] == 0
        assert "message" in result and "threshold" in result["message"].lower()

    def test_prune_auto_mode_with_explicit_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Auto mode with an explicit threshold prunes worse-than-threshold
        completed nodes."""
        from crucible.mcp.tools import tree_create, tree_prune
        from crucible.researcher.search_tree import SearchTree

        _patch_config(monkeypatch, tmp_path)
        tree_create({
            "name": "auto-th",
            "roots": [{"name": "good", "config": {"LR": "3e-4"}}],
        })

        # Add a "bad" node and complete both with metrics.
        tree_dir = tmp_path / ".crucible" / "search_trees" / "auto-th"
        tree = SearchTree.load(tree_dir)
        good_id = tree.meta["root_node_ids"][0]
        tree.record_result(good_id, {"val_loss": 1.0})  # below threshold
        # Expand a sibling root: not natively supported; instead expand a child
        # of good with a "bad" result.
        child_ids = tree.expand_node(good_id, [{"name": "bad", "config": {}, "hypothesis": ""}])
        bad_id = child_ids[0]
        tree.record_result(bad_id, {"val_loss": 99.0})  # well above threshold

        result = tree_prune({"name": "auto-th", "mode": "auto", "threshold": 5.0})
        assert result["mode"] == "auto"
        assert result["pruned_count"] >= 1
        assert bad_id in result["pruned_node_ids"]

    def test_prune_unknown_mode_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from crucible.core.errors import CrucibleError
        from crucible.mcp.tools import tree_create, tree_prune

        _patch_config(monkeypatch, tmp_path)
        tree_create({"name": "bad-mode"})

        with pytest.raises(CrucibleError, match="unknown mode"):
            tree_prune({"name": "bad-mode", "mode": "banana"})


class TestContextPushFindingAutoPromote:
    """Phase 1.6 auto-promote findings — review-driven test coverage."""

    def test_auto_promote_false_records_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Default: auto_promote omitted → records to state only, no hub call."""
        from crucible.mcp.tools import context_push_finding

        _patch_config(monkeypatch, tmp_path)
        result = context_push_finding({
            "finding": "test finding",
            "confidence": 0.95,
        })
        assert result["status"] == "recorded"
        assert "promoted" not in result

    def test_auto_promote_below_threshold_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """auto_promote=True but confidence below the promotion-rule
        threshold (0.6 for project→track) → records + auto_promote_skipped."""
        from crucible.mcp.tools import context_push_finding

        _patch_config(monkeypatch, tmp_path)
        result = context_push_finding({
            "finding": "weak signal",
            "confidence": 0.3,  # well below 0.6 threshold for project→track
            "auto_promote": True,
        })
        assert result["status"] == "recorded"
        assert result.get("auto_promote_skipped") is True
        assert "0.30" in result["reason"] or "below threshold" in result["reason"]

    def test_auto_promote_global_scope_has_high_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Phase 1.6 review fix: ('project', 'global') key was missing from
        PROMOTION_RULES, so auto_promote_scope='global' silently bypassed
        the confidence gate. Now there's an explicit 0.9 threshold."""
        from crucible.core.finding import PROMOTION_RULES
        # Confirm the rule exists with a high bar.
        assert ("project", "global") in PROMOTION_RULES
        assert PROMOTION_RULES[("project", "global")]["min_confidence"] >= 0.9

    def test_h13_string_confidence_coerced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """H.1.3 fix: an LLM-supplied "0.85" string used to crash the
        promotion-rule comparison. Now coerced to float at the entry."""
        from crucible.mcp.tools import context_push_finding

        _patch_config(monkeypatch, tmp_path)
        result = context_push_finding({
            "finding": "string-conf test",
            "confidence": "0.85",  # string, not float
        })
        assert result["status"] == "recorded"
        assert isinstance(result["entry"]["confidence"], float)
        assert abs(result["entry"]["confidence"] - 0.85) < 1e-9

    def test_h13_bad_confidence_surfaces_clean_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """H.1.3 fix: an unparseable confidence value surfaces a typed
        error instead of leaking ValueError."""
        from crucible.mcp.tools import context_push_finding

        _patch_config(monkeypatch, tmp_path)
        result = context_push_finding({
            "finding": "bad-conf",
            "confidence": "not-a-number",
        })
        assert "error" in result
        assert "confidence" in result["error"]
        assert "must be a number" in result["error"]

    def test_h13_concurrent_pushes_under_write_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """H.1.3: prior to the write_lock fix, two concurrent
        context_push_finding calls could lose a finding via the
        unsynchronized load → add → save sequence. Smoke-test that
        the locked path accepts back-to-back calls without dropping
        the count (full multiprocess test would be heavier — this is
        an in-process invariant check)."""
        from crucible.mcp.tools import context_push_finding
        from crucible.researcher.state import ResearchState

        _patch_config(monkeypatch, tmp_path)
        for i in range(5):
            context_push_finding({
                "finding": f"f{i}",
                "confidence": 0.5,
            })

        state = ResearchState(tmp_path / "research_state.jsonl")
        assert len(state.findings) == 5, (
            f"expected 5 findings persisted; got {len(state.findings)}"
        )

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
    @staticmethod
    def _patch_queue_contract(monkeypatch):
        monkeypatch.setattr(
            "crucible.mcp.tools._queue_contract_fields",
            lambda config: {"execution_provider": "runpod", "contract_status": "valid", "wandb": {"project": "test"}},
        )

    def test_enqueues_pending_nodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_enqueue_pending

        _patch_config(monkeypatch, tmp_path)
        self._patch_queue_contract(monkeypatch)

        create_res = tree_create({
            "name": "enq-tree",
            "roots": [
                {"name": "exp-1", "config": {"LR": "1e-4"}},
                {"name": "exp-2", "config": {"LR": "3e-4"}},
            ],
        })

        call_log: list[Any] = []

        def fake_enqueue(queue_path, experiments, limit=0):
            call_log.append(experiments)
            return [
                {"experiment_name": e["name"], "run_id": f"run_{e['name']}"}
                for e in experiments
            ]

        monkeypatch.setattr(
            "crucible.fleet.queue.enqueue_experiments",
            fake_enqueue,
        )

        result = tree_enqueue_pending({"name": "enq-tree"})
        assert result["status"] == "enqueued"
        assert result["enqueued"] == 2
        assert len(result["items"]) == 2

    def test_no_pending_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_create, tree_enqueue_pending, tree_prune

        _patch_config(monkeypatch, tmp_path)
        self._patch_queue_contract(monkeypatch)

        create_res = tree_create({
            "name": "nopend-tree",
            "roots": [{"name": "r", "config": {}}],
        })
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


# ---------------------------------------------------------------------------
# tree_auto_expand (Phase 1.3) — action-based contract, no LLM keys in Crucible
# ---------------------------------------------------------------------------


def _make_expandable_tree(tmp_path: Path) -> tuple[str, str]:
    """Create a tree with a single root that's been completed and is expandable.
    Returns (tree_name, root_node_id)."""
    from crucible.mcp.tools import tree_create
    from crucible.researcher.search_tree import SearchTree

    res = tree_create({
        "name": "ae-tree",
        "roots": [{"name": "root", "config": {"LR": "3e-4"}}],
    })
    root_id = res["root_node_ids"][0]
    tree_dir = tmp_path / ".crucible" / "search_trees" / "ae-tree"
    tree = SearchTree.load(tree_dir)
    tree.record_result(root_id, {"val_loss": 2.0})
    return "ae-tree", root_id


class TestSearchTreeSnapshot:
    def test_snapshot_shape(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        from crucible.researcher.search_tree import SearchTree
        tree = SearchTree.load(tmp_path / ".crucible" / "search_trees" / name)

        snap = tree.snapshot(node_id=root_id)
        assert snap["tree_name"] == name
        assert snap["node_id"] == root_id
        assert snap["total_nodes"] == 1
        assert "content_hash" in snap and len(snap["content_hash"]) == 16

    def test_snapshot_changes_when_node_added(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        from crucible.researcher.search_tree import SearchTree
        tree = SearchTree.load(tmp_path / ".crucible" / "search_trees" / name)

        before = tree.snapshot(node_id=root_id)
        tree.expand_node(root_id, [{"name": "child", "config": {}, "hypothesis": ""}])
        after = tree.snapshot(node_id=root_id)
        assert before != after
        assert before["content_hash"] != after["content_hash"]

    def test_snapshot_changes_when_result_recorded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A new result updates best_metric, which the snapshot tracks."""
        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        from crucible.researcher.search_tree import SearchTree
        tree = SearchTree.load(tmp_path / ".crucible" / "search_trees" / name)
        new_ids = tree.expand_node(root_id, [{"name": "c", "config": {}, "hypothesis": ""}])
        child_id = new_ids[0]

        before = tree.snapshot(node_id=root_id)
        tree.record_result(child_id, {"val_loss": 1.0})  # beats root's 2.0
        after = tree.snapshot(node_id=root_id)
        assert before["content_hash"] != after["content_hash"]

    def test_snapshot_tracks_parent_children_field(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Regression test: snapshot reads the node 'children' field, not the
        non-existent 'child_node_ids'. The original Phase 1.3 commit had the
        wrong field name; this test pins the parent's child-list contribution
        to the content hash."""
        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        from crucible.researcher.search_tree import SearchTree
        tree = SearchTree.load(tmp_path / ".crucible" / "search_trees" / name)

        # Confirm the field name on a real node, then verify the snapshot
        # changes when that field grows.
        root_node = tree.get_node(root_id)
        assert "children" in root_node, "schema invariant: nodes have 'children' key"

        before = tree.snapshot(node_id=root_id)
        tree.expand_node(root_id, [
            {"name": "c1", "config": {}, "hypothesis": ""},
            {"name": "c2", "config": {}, "hypothesis": ""},
        ])
        after = tree.snapshot(node_id=root_id)
        assert before["content_hash"] != after["content_hash"]
        # Pinpoint: the root node's children list grew; that's now reflected
        # in the per-node tuples that compose the content hash.
        assert len(tree.get_node(root_id)["children"]) == 2


class TestTreeAutoExpandRequestPrompt:
    def test_returns_prompt_schema_and_snapshot(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({
            "action": "request_prompt",
            "name": name,
            "node_id": root_id,
            "n_children": 2,
        })
        assert out["action"] == "request_prompt"
        assert out["stage"] == "tree_auto_expand"
        assert isinstance(out["system"], str) and len(out["system"]) > 0
        assert isinstance(out["user"], str) and "LR" in out["user"]
        assert out["schema"]["type"] == "array"
        assert out["tree_snapshot"]["tree_name"] == name
        assert out["tree_snapshot"]["node_id"] == root_id
        assert out["n_children"] == 2

    def test_unknown_node_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        _make_expandable_tree(tmp_path)

        out = tree_auto_expand({
            "action": "request_prompt",
            "name": "ae-tree",
            "node_id": "no-such-node",
        })
        assert "error" in out and "not found" in out["error"]


class TestTreeAutoExpandSubmit:
    def test_applies_response(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        prompt = tree_auto_expand({
            "action": "request_prompt", "name": name, "node_id": root_id, "n_children": 2,
        })
        response = [
            {"name": "low_lr", "config": {"LR": "1e-4"}, "hypothesis": "lower"},
            {"name": "high_lr", "config": {"LR": "1e-3"}, "hypothesis": "higher"},
        ]
        out = tree_auto_expand({
            "action": "submit",
            "name": name,
            "node_id": root_id,
            "response": response,
            "tree_snapshot": prompt["tree_snapshot"],
        })
        assert out["action"] == "submit"
        assert out["status"] == "auto_expanded"
        assert len(out["new_node_ids"]) == 2
        assert out["total_nodes"] == 3

    def test_submit_accepts_json_string_response(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({
            "action": "submit",
            "name": name,
            "node_id": root_id,
            "response": json.dumps([
                {"name": "a", "config": {"LR": "1e-4"}, "hypothesis": ""},
            ]),
        })
        assert out["status"] == "auto_expanded"
        assert len(out["new_node_ids"]) == 1

    def test_submit_accepts_dict_with_children_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({
            "action": "submit",
            "name": name,
            "node_id": root_id,
            "response": {"children": [
                {"name": "a", "config": {"LR": "1e-4"}, "hypothesis": ""},
            ]},
        })
        assert out["status"] == "auto_expanded"

    def test_submit_rejects_stale_tree_snapshot(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """If the tree advanced between request_prompt and submit, raise."""
        from crucible.core.errors import StaleSubmitError
        from crucible.mcp.tools import tree_auto_expand
        from crucible.researcher.search_tree import SearchTree

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        prompt = tree_auto_expand({
            "action": "request_prompt", "name": name, "node_id": root_id,
        })
        stale = prompt["tree_snapshot"]

        # Simulate a peer process: load tree, expand, save (expand_node persists).
        peer = SearchTree.load(tmp_path / ".crucible" / "search_trees" / name)
        peer.expand_node(root_id, [{"name": "peer", "config": {}, "hypothesis": ""}])

        with pytest.raises(StaleSubmitError, match="advanced"):
            tree_auto_expand({
                "action": "submit",
                "name": name,
                "node_id": root_id,
                "response": [{"name": "x", "config": {}, "hypothesis": ""}],
                "tree_snapshot": stale,
            })

    def test_submit_without_snapshot_skips_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Snapshot is opt-in for backward compat."""
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({
            "action": "submit",
            "name": name,
            "node_id": root_id,
            "response": [{"name": "x", "config": {}, "hypothesis": ""}],
        })
        assert out["status"] == "auto_expanded"

    def test_submit_rejects_dict_with_unrecognized_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """G.4 seam 3: a dict response with no recognized key (e.g.,
        orchestrator typed ``results`` instead of ``children``) must
        surface an error, not silently expand zero children. The
        dispatcher wraps CrucibleError as {"error": ...} for
        MCP-friendliness, so we assert on that shape."""
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({
            "action": "submit",
            "name": name,
            "node_id": root_id,
            "response": {"results": [
                {"name": "x", "config": {}, "hypothesis": ""},
            ]},
        })
        assert "error" in out
        assert "children" in out["error"].lower() or "no 'children'" in out["error"]
        # And no children were actually created.
        assert "status" not in out


class TestTreeAutoExpandDispatch:
    def test_missing_action_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """No action is an error pointing at the request_prompt/submit contract."""
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({"name": name, "node_id": root_id})
        assert "error" in out
        assert "action" in out["error"].lower()
        assert "request_prompt" in out["error"]

    def test_unknown_action_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.mcp.tools import tree_auto_expand

        _patch_config(monkeypatch, tmp_path)
        name, root_id = _make_expandable_tree(tmp_path)

        out = tree_auto_expand({"action": "banana", "name": name, "node_id": root_id})
        assert "error" in out and "unknown action" in out["error"].lower()
