"""Tests for GRPO-style group-relative advantage + top-K selection.

GIANTS uses GRPO (Group Relative Policy Optimization) over candidate
batches: sample N candidates, score each with the eval judge, normalize
scores within the group to advantages, keep the top-K. We expose this
as a tree-expansion policy alongside UCB1 / greedy / epsilon-greedy.

Pure functions; orchestrator scores candidates externally and passes
``judge_score`` per candidate. No internal LLM call.
"""
from __future__ import annotations

import math

import pytest

from crucible.researcher.grpo import (
    compute_advantages,
    select_top_k,
)


class TestComputeAdvantages:
    def test_z_score_normalizes_to_zero_mean(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        advantages = compute_advantages(scores, normalization="z_score")
        assert math.isclose(sum(advantages), 0.0, abs_tol=1e-9)

    def test_z_score_unit_variance(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        advantages = compute_advantages(scores, normalization="z_score")
        n = len(advantages)
        var = sum(a * a for a in advantages) / n
        assert math.isclose(var, 1.0, rel_tol=1e-6)

    def test_z_score_all_equal_returns_zeros(self) -> None:
        scores = [3.0, 3.0, 3.0]
        advantages = compute_advantages(scores, normalization="z_score")
        assert advantages == [0.0, 0.0, 0.0]

    def test_min_max_scales_to_unit_range(self) -> None:
        scores = [0.0, 5.0, 10.0]
        advantages = compute_advantages(scores, normalization="min_max")
        assert min(advantages) == 0.0
        assert max(advantages) == 1.0
        assert math.isclose(advantages[1], 0.5, abs_tol=1e-9)

    def test_min_max_all_equal_returns_zeros(self) -> None:
        advantages = compute_advantages([2.0, 2.0, 2.0], normalization="min_max")
        assert advantages == [0.0, 0.0, 0.0]

    def test_invalid_normalization_rejected(self) -> None:
        with pytest.raises(ValueError, match="normalization"):
            compute_advantages([1.0, 2.0], normalization="bogus")

    def test_empty_returns_empty(self) -> None:
        assert compute_advantages([], normalization="z_score") == []

    def test_higher_score_yields_higher_advantage(self) -> None:
        scores = [0.5, 0.8, 0.2, 0.9]
        advantages = compute_advantages(scores, normalization="z_score")
        # Index 3 has the highest score → highest advantage.
        assert advantages[3] == max(advantages)
        assert advantages[2] == min(advantages)


class TestSelectTopK:
    def test_picks_top_k_by_score(self) -> None:
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        idx = select_top_k(scores, top_k=2)
        assert sorted(idx) == [1, 4]

    def test_top_k_larger_than_pool_returns_all(self) -> None:
        scores = [0.1, 0.2, 0.3]
        idx = select_top_k(scores, top_k=10)
        assert sorted(idx) == [0, 1, 2]

    def test_top_k_zero_returns_empty(self) -> None:
        assert select_top_k([1.0, 2.0], top_k=0) == []

    def test_stable_ordering_for_ties(self) -> None:
        # When ties occur, lower-index wins → deterministic.
        scores = [0.5, 0.5, 0.5]
        idx = select_top_k(scores, top_k=2)
        assert idx == [0, 1]


class TestGrpoIntegration:
    def test_full_pipeline_picks_best(self) -> None:
        candidates = [
            {"name": "a", "config": {"X": "1"}},
            {"name": "b", "config": {"X": "2"}},
            {"name": "c", "config": {"X": "3"}},
            {"name": "d", "config": {"X": "4"}},
        ]
        scores = [3.0, 1.0, 4.0, 2.0]

        advantages = compute_advantages(scores, normalization="z_score")
        idx = select_top_k(advantages, top_k=2)

        kept = [(candidates[i]["name"], advantages[i]) for i in idx]
        names = [k[0] for k in kept]
        assert "c" in names  # highest raw score → highest advantage → kept
        assert "a" in names  # second-highest
