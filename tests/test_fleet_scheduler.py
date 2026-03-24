"""Tests for crucible.fleet.scheduler — dispatch, collection, merge."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crucible.fleet.scheduler import (
    dispatch,
    merge_results,
    launch_experiment,
    _interpolate_baseline,
)


def _make_node(name: str, state: str = "ready") -> dict[str, Any]:
    ready = state == "ready"
    return {
        "name": name,
        "node_id": f"id_{name}",
        "state": state,
        "env_ready": ready,
        "dataset_ready": ready,
        "git_sha": "abc" if ready else None,
        "ssh_host": f"10.0.0.{hash(name) % 256}",
        "ssh_port": 22,
        "user": "root",
        "workspace_path": "/workspace/project",
        "python_bin": "python3",
    }


def _make_queue_item(
    name: str,
    lease_state: str = "queued",
    assigned_node: str | None = None,
    priority: int = 0,
    created_at: str = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "run_id": f"run_{name}",
        "experiment_name": name,
        "backend": "torch",
        "tier": "smoke",
        "config": {"LR": "0.001"},
        "tags": [],
        "lease_state": lease_state,
        "assigned_node": assigned_node,
        "priority": priority,
        "created_at": created_at,
        "wave": "wave1",
    }


class TestDispatch:
    @patch("crucible.fleet.scheduler.remote_exec")
    def test_assigns_to_idle_nodes(self, mock_exec, tmp_path: Path):
        """Queued items get assigned to ready idle nodes."""
        mock_exec.return_value = MagicMock(stdout="12345\n")
        nodes = [_make_node("gpu-1"), _make_node("gpu-2")]
        queue = [_make_queue_item("exp_a"), _make_queue_item("exp_b")]
        queue_path = tmp_path / "queue.jsonl"

        result = dispatch(
            nodes, queue, queue_path=queue_path, max_assignments=10,
        )
        assigned = [r for r in result if r["lease_state"] == "running"]
        assert len(assigned) == 2
        assert assigned[0]["assigned_node"] == "gpu-1"
        assert assigned[1]["assigned_node"] == "gpu-2"

    @patch("crucible.fleet.scheduler.remote_exec")
    def test_skips_busy_nodes(self, mock_exec, tmp_path: Path):
        """Nodes with running experiments are not double-assigned."""
        mock_exec.return_value = MagicMock(stdout="12345\n")
        nodes = [_make_node("gpu-1")]
        queue = [
            _make_queue_item("exp_running", lease_state="running", assigned_node="gpu-1"),
            _make_queue_item("exp_queued"),
        ]
        queue_path = tmp_path / "queue.jsonl"

        result = dispatch(
            nodes, queue, queue_path=queue_path, max_assignments=10,
        )
        # gpu-1 is busy, so exp_queued stays queued
        queued = [r for r in result if r["lease_state"] == "queued"]
        assert len(queued) == 1
        assert queued[0]["experiment_name"] == "exp_queued"

    @patch("crucible.fleet.scheduler.remote_exec")
    def test_respects_max_assignments(self, mock_exec, tmp_path: Path):
        """Only max_assignments items are dispatched."""
        mock_exec.return_value = MagicMock(stdout="12345\n")
        nodes = [_make_node("gpu-1"), _make_node("gpu-2")]
        queue = [_make_queue_item("exp_a"), _make_queue_item("exp_b")]
        queue_path = tmp_path / "queue.jsonl"

        result = dispatch(
            nodes, queue, queue_path=queue_path, max_assignments=1,
        )
        running = [r for r in result if r["lease_state"] == "running"]
        assert len(running) == 1


    @patch("crucible.fleet.scheduler.remote_exec")
    def test_dispatches_high_priority_first(self, mock_exec, tmp_path: Path):
        """Higher priority items dispatch before lower, regardless of queue order."""
        mock_exec.return_value = MagicMock(stdout="12345\n")
        nodes = [_make_node("gpu-1")]
        queue = [
            _make_queue_item("low_prio", priority=0, created_at="2026-01-01T00:00:00Z"),
            _make_queue_item("high_prio", priority=10, created_at="2026-01-02T00:00:00Z"),
        ]
        queue_path = tmp_path / "queue.jsonl"

        result = dispatch(
            nodes, queue, queue_path=queue_path, max_assignments=1,
        )
        running = [r for r in result if r["lease_state"] == "running"]
        assert len(running) == 1
        assert running[0]["experiment_name"] == "high_prio"

    @patch("crucible.fleet.scheduler.remote_exec")
    def test_skips_finished_items_in_queue(self, mock_exec, tmp_path: Path):
        """Completed/failed items in the queue don't block dispatch."""
        mock_exec.return_value = MagicMock(stdout="12345\n")
        nodes = [_make_node("gpu-1")]
        queue = [
            _make_queue_item("old_done", lease_state="completed"),
            _make_queue_item("old_failed", lease_state="failed"),
            _make_queue_item("new_exp"),
        ]
        queue_path = tmp_path / "queue.jsonl"

        result = dispatch(
            nodes, queue, queue_path=queue_path, max_assignments=5,
        )
        running = [r for r in result if r["lease_state"] == "running"]
        assert len(running) == 1
        assert running[0]["experiment_name"] == "new_exp"


class TestMergeResults:
    def test_merges_per_node_files(self, tmp_path: Path):
        """merge_results combines per-node result files into one JSONL."""
        fleet_runs = tmp_path / "fleet_runs"
        (fleet_runs / "gpu-1").mkdir(parents=True)
        (fleet_runs / "gpu-2").mkdir(parents=True)

        results_1 = [{"id": "run_a", "status": "completed", "result": {"val_bpb": 1.5}}]
        results_2 = [{"id": "run_b", "status": "completed", "result": {"val_bpb": 1.4}}]

        with open(fleet_runs / "gpu-1" / "experiments.jsonl", "w") as f:
            for r in results_1:
                f.write(json.dumps(r) + "\n")
        with open(fleet_runs / "gpu-2" / "experiments.jsonl", "w") as f:
            for r in results_2:
                f.write(json.dumps(r) + "\n")

        merged_file = tmp_path / "fleet_results.jsonl"
        merged_file.write_text("")  # start empty

        merge_results(fleet_runs, merged_file)

        from crucible.core.io import read_jsonl
        merged = list(read_jsonl(merged_file))
        assert len(merged) == 2
        ids = {r["id"] for r in merged}
        assert ids == {"run_a", "run_b"}

    def test_deduplicates_by_id(self, tmp_path: Path):
        """merge_results deduplicates by run ID, keeping latest."""
        fleet_runs = tmp_path / "fleet_runs"
        (fleet_runs / "gpu-1").mkdir(parents=True)

        results = [{"id": "run_a", "status": "completed", "result": {"val_bpb": 1.5}}]
        with open(fleet_runs / "gpu-1" / "experiments.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        merged_file = tmp_path / "fleet_results.jsonl"
        # Pre-populate with same ID
        with open(merged_file, "w") as f:
            f.write(json.dumps({"id": "run_a", "status": "partial", "result": {"val_bpb": 2.0}}) + "\n")

        merge_results(fleet_runs, merged_file)

        from crucible.core.io import read_jsonl
        merged = list(read_jsonl(merged_file))
        assert len(merged) == 1
        assert merged[0]["result"]["val_bpb"] == 1.5  # node file wins (latest)


class TestInterpolateBaseline:
    def test_interpolates_between_points(self):
        curve = [(0, 10.0), (100, 5.0), (200, 2.0)]
        result = _interpolate_baseline(curve, 50)
        assert result == pytest.approx(7.5)

    def test_clamps_at_boundaries(self):
        curve = [(100, 5.0), (200, 2.0)]
        assert _interpolate_baseline(curve, 50) == 5.0
        assert _interpolate_baseline(curve, 300) == 2.0

    def test_empty_curve(self):
        assert _interpolate_baseline([], 100) is None
