"""Tests for crucible.researcher.batch_design."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crucible.researcher.batch_design import design_batch, DEFAULT_TIER_COSTS
from crucible.researcher.state import ResearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(tmp_path: Path, budget_hours: float = 10.0) -> ResearchState:
    return ResearchState(tmp_path / "state.jsonl", budget_hours=budget_hours)


def _make_hypothesis(
    name: str,
    config: dict[str, str] | None = None,
    confidence: float = 0.5,
    family: str = "test",
) -> dict[str, Any]:
    return {
        "name": name,
        "hypothesis": f"Testing {name}",
        "config": config if config is not None else {"LR": "0.001"},
        "confidence": confidence,
        "family": family,
        "rationale": f"Rationale for {name}",
    }


# ---------------------------------------------------------------------------
# design_batch
# ---------------------------------------------------------------------------

class TestDesignBatch:
    def test_includes_baseline(self, tmp_path):
        state = _make_state(tmp_path)
        hyps = [_make_hypothesis("exp1")]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=1,
            baseline_config={"LR": "0.001"},
        )
        names = [exp["name"] for exp in batch]
        assert "baseline_control" in names

    def test_baseline_excluded_if_no_config(self, tmp_path):
        state = _make_state(tmp_path)
        hyps = [_make_hypothesis("exp1")]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=1,
            baseline_config=None,
        )
        names = [exp["name"] for exp in batch]
        assert "baseline_control" not in names

    def test_baseline_excluded_if_recently_run(self, tmp_path):
        state = _make_state(tmp_path)
        # Add a recent baseline to history
        state.history.append({"experiment": {"name": "baseline_control"}, "result": {}})
        hyps = [_make_hypothesis("exp1")]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=1,
            baseline_config={"LR": "0.001"},
        )
        names = [exp["name"] for exp in batch]
        assert "baseline_control" not in names

    def test_hypothesis_conversion(self, tmp_path):
        state = _make_state(tmp_path)
        hyps = [
            _make_hypothesis("exp_a", config={"LR": "0.002"}, confidence=0.8, family="lr"),
        ]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=3,
            baseline_config=None,
        )
        assert len(batch) == 1
        exp = batch[0]
        assert exp["name"] == "exp_a"
        assert exp["config"]["LR"] == "0.002"
        assert exp["tier"] == "proxy"
        assert exp["backend"] == "torch"
        assert "autonomous" in exp["tags"]
        assert "lr" in exp["tags"]
        assert "iter_3" in exp["tags"]
        assert exp["wave"] == "auto_iter_3"
        assert exp["priority"] == 80  # confidence * 100

    def test_empty_config_hypothesis_skipped(self, tmp_path):
        state = _make_state(tmp_path)
        hyps = [
            _make_hypothesis("empty", config={}),
            _make_hypothesis("valid", config={"LR": "0.001"}),
        ]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=1,
            baseline_config=None,
        )
        names = [exp["name"] for exp in batch]
        assert "empty" not in names
        assert "valid" in names

    def test_budget_constraint(self, tmp_path):
        # Budget of 1.0 hours with proxy cost of 0.5 hours each
        state = _make_state(tmp_path, budget_hours=1.0)
        hyps = [
            _make_hypothesis(f"exp_{i}", config={"LR": str(i)})
            for i in range(10)
        ]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=1,
            baseline_config=None,
        )
        # With 1.0 budget and 0.5 cost per experiment, should fit at most 2
        assert len(batch) <= 2

    def test_custom_tier_costs(self, tmp_path):
        state = _make_state(tmp_path, budget_hours=1.0)
        hyps = [
            _make_hypothesis(f"exp_{i}", config={"LR": str(i)})
            for i in range(10)
        ]
        batch = design_batch(
            hyps, state, tier="cheap", backend="torch", iteration=1,
            tier_costs={"cheap": 0.1},
            baseline_config=None,
        )
        # 1.0 budget / 0.1 cost = 10 possible, but budget check is
        # state.budget_remaining < tier_cost * (len(batch) + 1)
        assert len(batch) >= 5

    def test_tags_include_iteration(self, tmp_path):
        state = _make_state(tmp_path)
        hyps = [_make_hypothesis("tagged")]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=7,
            baseline_config=None,
        )
        assert "iter_7" in batch[0]["tags"]

    def test_wave_name(self, tmp_path):
        state = _make_state(tmp_path)
        hyps = [_make_hypothesis("wave_test")]
        batch = design_batch(
            hyps, state, tier="proxy", backend="torch", iteration=5,
            baseline_config=None,
        )
        assert batch[0]["wave"] == "auto_iter_5"

    def test_default_tier_costs_keys(self):
        assert "smoke" in DEFAULT_TIER_COSTS
        assert "proxy" in DEFAULT_TIER_COSTS
        assert "medium" in DEFAULT_TIER_COSTS
        assert "promotion" in DEFAULT_TIER_COSTS
