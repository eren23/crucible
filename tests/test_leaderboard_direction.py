"""Tests for leaderboard direction-aware sorting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import MetricsConfig, ProjectConfig
from crucible.analysis.leaderboard import leaderboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    name: str,
    metric_value: float,
    metric_key: str = "val_loss",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "completed",
        "result": {metric_key: metric_value},
        "config": {},
    }


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Direction tests
# ---------------------------------------------------------------------------


class TestLeaderboardDirection:
    def test_minimize_sorts_ascending(self, tmp_path):
        """When direction='minimize', lowest metric first (existing behavior)."""
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
            metrics=MetricsConfig(primary="val_loss", direction="minimize"),
        )
        results = [
            _make_result("worst", 3.0),
            _make_result("best", 1.0),
            _make_result("mid", 2.0),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(top_n=10, cfg=cfg)
        assert ranked[0]["name"] == "best"
        assert ranked[1]["name"] == "mid"
        assert ranked[2]["name"] == "worst"

    def test_maximize_sorts_descending(self, tmp_path):
        """When direction='maximize', highest metric first."""
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
            metrics=MetricsConfig(primary="accuracy", direction="maximize"),
        )
        results = [
            _make_result("worst", 0.70, "accuracy"),
            _make_result("best", 0.95, "accuracy"),
            _make_result("mid", 0.85, "accuracy"),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(top_n=10, cfg=cfg)
        assert ranked[0]["name"] == "best"
        assert ranked[1]["name"] == "mid"
        assert ranked[2]["name"] == "worst"

    def test_explicit_direction_overrides_config(self, tmp_path):
        """The direction= kwarg should override config direction."""
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
            metrics=MetricsConfig(primary="val_loss", direction="minimize"),
        )
        results = [
            _make_result("a", 1.0),
            _make_result("b", 3.0),
            _make_result("c", 2.0),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        # Override to maximize
        ranked = leaderboard(top_n=10, direction="maximize", cfg=cfg)
        assert ranked[0]["name"] == "b"  # highest
        assert ranked[2]["name"] == "a"  # lowest

    def test_default_direction_is_minimize(self, tmp_path):
        """Without config or explicit direction, should default to minimize."""
        results = [
            _make_result("a", 3.0),
            _make_result("b", 1.0),
            _make_result("c", 2.0),
        ]
        # Pass results directly, no cfg
        ranked = leaderboard(results, top_n=10, metric="val_loss")
        assert ranked[0]["name"] == "b"
        assert ranked[2]["name"] == "a"

    def test_maximize_with_top_n(self, tmp_path):
        """Top N should return the N best when direction=maximize."""
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
            metrics=MetricsConfig(primary="reward", direction="maximize"),
        )
        results = [_make_result(f"exp_{i}", float(i), "reward") for i in range(10)]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(top_n=3, cfg=cfg)
        assert len(ranked) == 3
        # Top 3 by highest reward: 9, 8, 7
        assert ranked[0]["name"] == "exp_9"
        assert ranked[1]["name"] == "exp_8"
        assert ranked[2]["name"] == "exp_7"
