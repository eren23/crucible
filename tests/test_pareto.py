"""Tests for N-dimensional Pareto frontier utilities and SearchTree integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.analysis.leaderboard import (
    dominates,
    hypervolume_2d,
    pareto_frontier_nd,
)
from crucible.researcher.search_tree import SearchTree


# ---------------------------------------------------------------------------
# Pure utility tests
# ---------------------------------------------------------------------------


def test_dominates_minimize() -> None:
    assert dominates([1.0, 1.0], [2.0, 2.0], ["minimize", "minimize"])
    assert not dominates([1.0, 1.0], [1.0, 1.0], ["minimize", "minimize"])
    assert not dominates([1.0, 3.0], [2.0, 2.0], ["minimize", "minimize"])


def test_dominates_mixed_directions() -> None:
    # Max/min combo: higher first, lower second dominates
    assert dominates([5.0, 1.0], [3.0, 2.0], ["maximize", "minimize"])
    assert not dominates([3.0, 1.0], [5.0, 2.0], ["maximize", "minimize"])


def test_dominates_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        dominates([1.0], [1.0, 2.0], ["minimize", "minimize"])


def test_pareto_frontier_nd_basic() -> None:
    # 2D: (1,4), (2,2), (4,1) trade off; (3,3) is dominated by (2,2).
    pts = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0]]
    dirs = ["minimize", "minimize"]
    idx = pareto_frontier_nd(pts, dirs)
    assert sorted(idx) == [0, 1, 2]


def test_pareto_frontier_nd_3d() -> None:
    pts = [
        [1.0, 3.0, 2.0],
        [2.0, 2.0, 2.0],
        [3.0, 1.0, 3.0],
        [2.5, 2.5, 2.5],  # Dominated by [2,2,2]
    ]
    dirs = ["minimize"] * 3
    idx = pareto_frontier_nd(pts, dirs)
    assert sorted(idx) == [0, 1, 2]


def test_pareto_frontier_nd_empty() -> None:
    assert pareto_frontier_nd([], ["minimize"]) == []


def test_hypervolume_2d_known() -> None:
    # Three non-dominated points, reference (5,5)
    # contributions:
    #   (1,4): (2-1)*(5-4)=1
    #   (2,2): (4-2)*(5-2)=6
    #   (4,1): (5-4)*(5-1)=4
    pts = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
    hv = hypervolume_2d(pts, ["minimize", "minimize"], reference=[5.0, 5.0])
    assert hv == pytest.approx(11.0)


def test_hypervolume_2d_with_maximize_axis() -> None:
    # Flip one axis; the frontier is the same set of points conceptually.
    pts = [[4.0, 4.0], [3.0, 2.0], [1.0, 1.0]]
    hv = hypervolume_2d(pts, ["maximize", "minimize"], reference=[0.0, 5.0])
    assert hv > 0.0


def test_hypervolume_2d_empty() -> None:
    assert hypervolume_2d([], ["minimize", "minimize"]) == 0.0


# ---------------------------------------------------------------------------
# SearchTree multi-metric integration
# ---------------------------------------------------------------------------


def _make_tree(tmp_path: Path, metrics: list[dict[str, str]] | None = None) -> SearchTree:
    tree_dir = tmp_path / "t"
    return SearchTree.create(
        tree_dir=tree_dir,
        name="demo",
        metrics=metrics,
    )


def test_search_tree_multi_metric_frontier(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path, metrics=[
        {"name": "val_bpb", "direction": "minimize"},
        {"name": "params", "direction": "minimize"},
    ])
    a = tree.add_root("a", {"X": "1"})
    b = tree.add_root("b", {"X": "2"})
    c = tree.add_root("c", {"X": "3"})

    # a and b trade off; c is dominated by both
    tree.record_result(a, {"val_bpb": 1.0, "params": 500})
    tree.record_result(b, {"val_bpb": 0.8, "params": 1000})
    tree.record_result(c, {"val_bpb": 1.2, "params": 1500})

    frontier = tree.pareto_nodes()
    assert a in frontier
    assert b in frontier
    assert c not in frontier


def test_search_tree_legacy_single_metric_fallback(tmp_path: Path) -> None:
    # Create without explicit metrics; primary_metric should drive frontier.
    tree = _make_tree(tmp_path)
    a = tree.add_root("a", {})
    b = tree.add_root("b", {})
    tree.record_result(a, {"val_bpb": 1.0})
    tree.record_result(b, {"val_bpb": 0.8})
    assert tree.pareto_nodes() == [b]


def test_frontier_summary_includes_best_per_metric(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path, metrics=[
        {"name": "accuracy", "direction": "maximize"},
        {"name": "tokens", "direction": "minimize"},
    ])
    a = tree.add_root("a", {})
    b = tree.add_root("b", {})
    tree.record_result(a, {"accuracy": 0.9, "tokens": 200})
    tree.record_result(b, {"accuracy": 0.8, "tokens": 100})

    summary = tree.frontier_summary()
    assert summary["frontier_size"] == 2
    assert summary["best_per_metric"]["accuracy"]["node_id"] == a
    assert summary["best_per_metric"]["tokens"]["node_id"] == b


def test_pareto_selection_policy(tmp_path: Path) -> None:
    tree = SearchTree.create(
        tree_dir=tmp_path / "tp",
        name="pareto_demo",
        expansion_policy="pareto",
        metrics=[
            {"name": "val_bpb", "direction": "minimize"},
            {"name": "params", "direction": "minimize"},
        ],
    )
    a = tree.add_root("a", {})
    b = tree.add_root("b", {})
    tree.record_result(a, {"val_bpb": 1.0, "params": 500})
    tree.record_result(b, {"val_bpb": 1.5, "params": 2000})  # dominated
    # Add pending children; frontier-ancestry child should come first
    tree.expand_node(a, [{"name": "a_child", "config": {"Y": "1"}}])
    tree.expand_node(b, [{"name": "b_child", "config": {"Y": "2"}}])

    picks = tree.select_next(n=2)
    # a is on the frontier; its child should rank ahead of b's.
    assert picks  # policy produces a non-empty selection
    first_node = tree.get_node(picks[0])
    assert first_node is not None
    assert first_node["parent_node_id"] == a


def test_candidate_storage_round_trip(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    nid = tree.add_root("c", {})
    path = tree.store_candidate(nid, "# candidate\n")
    assert path.exists()
    assert tree.load_candidate(nid) == "# candidate\n"
    # Config was augmented with HARNESS_CANDIDATE_ID and HARNESS_CANDIDATES_DIR
    cfg = tree.get_node(nid)["config"]
    assert cfg["HARNESS_CANDIDATE_ID"] == nid
    assert "HARNESS_CANDIDATES_DIR" in cfg


def test_candidate_id_safety(tmp_path: Path) -> None:
    from crucible.core.errors import SearchTreeError

    tree = _make_tree(tmp_path)
    nid = tree.add_root("ok", {})
    # Valid id works; path traversal / spaces should be rejected.
    tree.store_candidate(nid, "pass")
    with pytest.raises(SearchTreeError):
        tree.store_candidate("../evil", "pass")
    with pytest.raises(SearchTreeError):
        tree.load_candidate("../../etc/passwd")


def test_candidate_size_cap(tmp_path: Path) -> None:
    from crucible.core.errors import SearchTreeError

    tree = _make_tree(tmp_path)
    nid = tree.add_root("big", {})
    huge = "a" * (257 * 1024)  # above 256KB cap
    with pytest.raises(SearchTreeError, match="exceeds"):
        tree.store_candidate(nid, huge)
