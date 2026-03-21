"""Tests for crucible.analysis.leaderboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from crucible.core.config import ProjectConfig, load_config
from crucible.analysis.leaderboard import (
    leaderboard,
    sensitivity_analysis,
    pareto_frontier,
    _metric_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str,
    val_bpb: float,
    model_bytes: int = 50000,
    status: str = "completed",
    config: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "result": {"val_bpb": val_bpb, "val_loss": val_bpb * 1.1},
        "model_bytes": model_bytes,
        "config": config or {"LR": "0.001"},
    }


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# _metric_value
# ---------------------------------------------------------------------------

class TestMetricValue:
    def test_extracts_metric(self):
        r = {"result": {"val_bpb": 1.5}}
        assert _metric_value(r, "val_bpb") == 1.5

    def test_missing_metric_raises(self):
        r = {"result": {}}
        with pytest.raises(KeyError):
            _metric_value(r, "val_bpb")


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------

class TestLeaderboard:
    def test_ranks_by_metric_ascending(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("worst", 2.0),
            _make_result("best", 1.0),
            _make_result("mid", 1.5),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(top_n=10, cfg=cfg)
        assert len(ranked) == 3
        assert ranked[0]["name"] == "best"
        assert ranked[1]["name"] == "mid"
        assert ranked[2]["name"] == "worst"

    def test_top_n_limits(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [_make_result(f"exp_{i}", 1.0 + i * 0.1) for i in range(10)]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(top_n=3, cfg=cfg)
        assert len(ranked) == 3

    def test_empty_results(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        _write_results(tmp_path / "experiments.jsonl", [])
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(cfg=cfg)
        assert ranked == []

    def test_filters_non_completed(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("completed_exp", 1.5, status="completed"),
            _make_result("failed_exp", 1.0, status="failed"),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(cfg=cfg)
        assert len(ranked) == 1
        assert ranked[0]["name"] == "completed_exp"

    def test_filters_missing_metric(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("has_metric", 1.5),
            {"name": "no_metric", "status": "completed", "result": {}, "config": {}},
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        ranked = leaderboard(cfg=cfg)
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_returns_varying_keys(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("a", 1.5, config={"LR": "0.001", "BATCH": "64"}),
            _make_result("b", 1.3, config={"LR": "0.002", "BATCH": "64"}),
            _make_result("c", 1.1, config={"LR": "0.003", "BATCH": "128"}),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        sa = sensitivity_analysis(cfg=cfg)
        assert "LR" in sa
        assert "BATCH" in sa

    def test_excludes_constant_keys(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("a", 1.5, config={"LR": "0.001", "CONSTANT": "same"}),
            _make_result("b", 1.3, config={"LR": "0.002", "CONSTANT": "same"}),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        sa = sensitivity_analysis(cfg=cfg)
        assert "LR" in sa
        assert "CONSTANT" not in sa

    def test_sorted_by_metric(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("a", 2.0, config={"LR": "0.001"}),
            _make_result("b", 1.0, config={"LR": "0.002"}),
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        sa = sensitivity_analysis(cfg=cfg)
        assert sa["LR"][0][1] <= sa["LR"][1][1]

    def test_empty_results(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        _write_results(tmp_path / "experiments.jsonl", [])
        _write_results(tmp_path / "fleet.jsonl", [])

        sa = sensitivity_analysis(cfg=cfg)
        assert sa == {}


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_pareto_optimal_points(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("small_good", 1.2, model_bytes=30000),
            _make_result("big_best", 1.0, model_bytes=80000),
            _make_result("mid_mid", 1.1, model_bytes=50000),
            _make_result("big_bad", 1.5, model_bytes=90000),   # dominated
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        pareto = pareto_frontier(cfg=cfg)
        pareto_names = {p["name"] for p in pareto}
        # big_best (1.0, 80000) -> first, min_size=80000
        # mid_mid (1.1, 50000) -> 50000 < 80000, Pareto
        # small_good (1.2, 30000) -> 30000 < 50000, Pareto
        # big_bad (1.5, 90000) -> 90000 > 30000, not Pareto
        assert "big_best" in pareto_names
        assert "mid_mid" in pareto_names
        assert "small_good" in pareto_names
        assert "big_bad" not in pareto_names

    def test_empty_results(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        _write_results(tmp_path / "experiments.jsonl", [])
        _write_results(tmp_path / "fleet.jsonl", [])

        pareto = pareto_frontier(cfg=cfg)
        assert pareto == []

    def test_single_result(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [_make_result("only", 1.5, model_bytes=50000)]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        pareto = pareto_frontier(cfg=cfg)
        assert len(pareto) == 1

    def test_missing_model_bytes_excluded(self, tmp_path):
        cfg = ProjectConfig(
            project_root=tmp_path,
            results_file="experiments.jsonl",
            fleet_results_file="fleet.jsonl",
        )
        results = [
            _make_result("with_bytes", 1.5, model_bytes=50000),
            {"name": "no_bytes", "status": "completed", "result": {"val_bpb": 1.0},
             "model_bytes": None, "config": {}},
        ]
        _write_results(tmp_path / "experiments.jsonl", results)
        _write_results(tmp_path / "fleet.jsonl", [])

        pareto = pareto_frontier(cfg=cfg)
        pareto_names = {p["name"] for p in pareto}
        assert "no_bytes" not in pareto_names
