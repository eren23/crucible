"""Tests for crucible.analysis.results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import ProjectConfig
from crucible.analysis.results import (
    load_results,
    merged_results,
    completed_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str,
    val_bpb: float = 1.5,
    status: str = "completed",
    model_bytes: int = 50000,
) -> dict[str, Any]:
    return {
        "id": f"run_{name}",
        "name": name,
        "status": status,
        "result": {"val_bpb": val_bpb} if status == "completed" else None,
        "model_bytes": model_bytes,
        "config": {"LR": "0.001"},
    }


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def _make_cfg(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        project_root=tmp_path,
        results_file="experiments.jsonl",
        fleet_results_file="fleet.jsonl",
    )


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

class TestLoadResults:
    def test_load_local(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [
            _make_result("a"), _make_result("b"),
        ])
        results = load_results(cfg, source="local")
        assert len(results) == 2

    def test_load_fleet(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "fleet.jsonl", [_make_result("fleet_1")])
        results = load_results(cfg, source="fleet")
        assert len(results) == 1
        assert results[0]["name"] == "fleet_1"

    def test_load_project_runs(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        project_runs = tmp_path / ".crucible" / "projects" / "runs.jsonl"
        _write_results(project_runs, [{
            "run_id": "proj_1",
            "project": "demo",
            "variant_name": "lewm_slim_48d_2e_2p",
            "status": "launched",
            "resolved_overrides": {"SLIM_DIM": "48"},
        }])
        results = load_results(cfg, source="project")
        assert len(results) == 1
        assert results[0]["id"] == "proj_1"
        assert results[0]["name"] == "lewm_slim_48d_2e_2p"
        assert results[0]["config"]["SLIM_DIM"] == "48"

    def test_load_missing_file_returns_empty(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        results = load_results(cfg, source="local")
        assert results == []


# ---------------------------------------------------------------------------
# merged_results
# ---------------------------------------------------------------------------

class TestMergedResults:
    def test_merges_local_and_fleet(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [_make_result("local_exp")])
        _write_results(tmp_path / "fleet.jsonl", [_make_result("fleet_exp")])
        _write_results(tmp_path / ".crucible" / "projects" / "runs.jsonl", [{
            "run_id": "project_run_1",
            "project": "demo",
            "variant_name": "project_exp",
            "status": "launched",
        }])

        merged = merged_results(cfg)
        names = {r["name"] for r in merged}
        assert "local_exp" in names
        assert "fleet_exp" in names
        assert "project_exp" in names

    def test_dedup_by_name(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [
            {**_make_result("shared", val_bpb=2.0), "id": "local_shared"},
        ])
        _write_results(tmp_path / ".crucible" / "projects" / "runs.jsonl", [
            {
                "run_id": "run_shared",
                "project": "demo",
                "variant_name": "shared",
                "status": "launched",
            },
        ])
        _write_results(tmp_path / "fleet.jsonl", [
            {"id": "run_shared", **_make_result("shared", val_bpb=1.5)},
        ])

        merged = merged_results(cfg)
        # Latest (fleet) should win since it comes after project in the list and shares the same run id.
        shared = [r for r in merged if r["name"] == "shared"]
        assert len(shared) == 2
        fleet = [r for r in shared if r["id"] == "run_shared"]
        assert len(fleet) == 1
        assert fleet[0]["result"]["val_bpb"] == 1.5

    def test_project_record_latest_write_wins(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / ".crucible" / "projects" / "runs.jsonl", [
            {"run_id": "proj_1", "project": "demo", "variant_name": "demo_a", "status": "launched"},
            {"run_id": "proj_1", "project": "demo", "variant_name": "demo_a", "status": "completed", "result": {"val_bpb": 1.1}},
        ])
        merged = merged_results(cfg)
        assert len(merged) == 1
        assert merged[0]["status"] == "completed"
        assert merged[0]["result"]["val_bpb"] == 1.1

    def test_empty_sources(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [])
        _write_results(tmp_path / "fleet.jsonl", [])
        merged = merged_results(cfg)
        assert merged == []


# ---------------------------------------------------------------------------
# completed_results
# ---------------------------------------------------------------------------

class TestCompletedResults:
    def test_filters_completed_only(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [
            _make_result("done", status="completed"),
            _make_result("fail", status="failed"),
            _make_result("timeout", status="timeout"),
        ])
        _write_results(tmp_path / "fleet.jsonl", [])

        completed = completed_results(cfg)
        assert len(completed) == 1
        assert completed[0]["name"] == "done"

    def test_requires_result_dict(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [
            {"name": "no_result", "status": "completed", "result": None, "config": {}},
            _make_result("has_result", status="completed"),
        ])
        _write_results(tmp_path / "fleet.jsonl", [])

        completed = completed_results(cfg)
        assert len(completed) == 1
        assert completed[0]["name"] == "has_result"

    def test_includes_fleet_by_default(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [
            _make_result("local"),
        ])
        _write_results(tmp_path / "fleet.jsonl", [
            _make_result("fleet"),
        ])

        completed = completed_results(cfg)
        names = {r["name"] for r in completed}
        assert "local" in names
        assert "fleet" in names

    def test_exclude_fleet(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [_make_result("local")])
        _write_results(tmp_path / "fleet.jsonl", [_make_result("fleet")])

        completed = completed_results(cfg, include_fleet=False)
        names = {r["name"] for r in completed}
        assert "local" in names
        assert "fleet" not in names

    def test_empty_returns_empty(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(tmp_path / "experiments.jsonl", [])
        _write_results(tmp_path / "fleet.jsonl", [])

        completed = completed_results(cfg)
        assert completed == []
