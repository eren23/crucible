"""Tests for crucible.fleet.queue."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.fleet.queue import (
    make_run_id,
    load_queue,
    save_queue,
    reset_queue,
    enqueue_experiments,
    reconcile_queue_with_results,
    summarize_queue,
    wave_rows,
    wave_result_rows,
    results_by_id,
)


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------

class TestMakeRunId:
    def test_contains_name_slug(self):
        run_id = make_run_id("my_experiment")
        assert "my_experiment" in run_id or "my-experiment" in run_id

    def test_unique_ids(self):
        id1 = make_run_id("test")
        id2 = make_run_id("test")
        assert id1 != id2

    def test_sanitizes_special_chars(self):
        run_id = make_run_id("exp with spaces & symbols!")
        # Should not contain spaces or special chars other than - and _
        slug_part = run_id.split("-", 2)[-1]  # Skip timestamp prefix
        for ch in slug_part:
            if ch == "-":
                continue
            # uuid part at end has hex chars
            assert ch.isalnum() or ch in {"-", "_"}

    def test_truncates_long_names(self):
        long_name = "a" * 100
        run_id = make_run_id(long_name)
        # The slug portion should be at most 32 chars
        assert len(run_id) < 100


# ---------------------------------------------------------------------------
# load_queue / save_queue / reset_queue
# ---------------------------------------------------------------------------

class TestQueuePersistence:
    def test_load_missing_file(self, tmp_path):
        assert load_queue(tmp_path / "queue.jsonl") == []

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        rows = [
            {"experiment_name": "exp1", "tier": "proxy", "lease_state": "queued"},
            {"experiment_name": "exp2", "tier": "smoke", "lease_state": "running"},
        ]
        save_queue(path, rows)
        loaded = load_queue(path)
        assert len(loaded) == 2
        assert loaded[0]["experiment_name"] == "exp1"

    def test_reset_queue(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        save_queue(path, [{"x": 1}])
        reset_queue(path)
        assert load_queue(path) == []


# ---------------------------------------------------------------------------
# enqueue_experiments
# ---------------------------------------------------------------------------

class TestEnqueueExperiments:
    def _make_exp(self, name: str, tier: str = "proxy") -> dict[str, Any]:
        return {
            "name": name,
            "tier": tier,
            "backend": "torch",
            "config": {"LR": "0.001"},
            "tags": ["test"],
            "priority": 50,
            "wave": "wave_1",
        }

    def test_enqueue_new_experiments(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        exps = [self._make_exp("exp1"), self._make_exp("exp2")]
        added = enqueue_experiments(path, exps)
        assert len(added) == 2
        queue = load_queue(path)
        assert len(queue) == 2

    def test_dedup_by_name_and_tier(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        exps = [self._make_exp("exp1")]
        enqueue_experiments(path, exps)
        # Try to add same experiment again
        added = enqueue_experiments(path, exps)
        assert len(added) == 0
        queue = load_queue(path)
        assert len(queue) == 1

    def test_same_name_different_tier_allowed(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        exps = [
            self._make_exp("exp1", tier="proxy"),
            self._make_exp("exp1", tier="smoke"),
        ]
        added = enqueue_experiments(path, exps)
        assert len(added) == 2

    def test_limit_parameter(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        exps = [self._make_exp(f"exp_{i}") for i in range(10)]
        added = enqueue_experiments(path, exps, limit=3)
        assert len(added) == 3
        queue = load_queue(path)
        assert len(queue) == 3

    def test_queue_item_fields(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        exps = [self._make_exp("check_fields")]
        added = enqueue_experiments(path, exps)
        item = added[0]
        assert item["experiment_name"] == "check_fields"
        assert item["tier"] == "proxy"
        assert item["backend"] == "torch"
        assert item["config"]["LR"] == "0.001"
        assert item["tags"] == ["test"]
        assert item["priority"] == 50
        assert item["wave"] == "wave_1"
        assert item["assigned_node"] is None
        assert item["lease_state"] == "queued"
        assert item["attempt"] == 0
        assert item["started_at"] is None
        assert item["ended_at"] is None
        assert item["result_status"] is None
        assert "run_id" in item
        assert "created_at" in item


# ---------------------------------------------------------------------------
# reconcile_queue_with_results
# ---------------------------------------------------------------------------

class TestReconcileQueueWithResults:
    def test_matches_by_run_id(self):
        rows = [
            {"run_id": "run_1", "lease_state": "running", "ended_at": None, "result_status": None},
            {"run_id": "run_2", "lease_state": "running", "ended_at": None, "result_status": None},
        ]
        result_index = {
            "run_1": {"status": "completed"},
        }
        updated = reconcile_queue_with_results(rows, result_index)
        assert updated[0]["result_status"] == "completed"
        assert updated[0]["lease_state"] == "completed"
        assert updated[0]["ended_at"] is not None
        assert updated[1]["result_status"] is None
        assert updated[1]["lease_state"] == "running"

    def test_failed_result_sets_finished(self):
        rows = [
            {"run_id": "run_1", "lease_state": "running", "ended_at": None, "result_status": None},
        ]
        result_index = {
            "run_1": {"status": "failed"},
        }
        updated = reconcile_queue_with_results(rows, result_index)
        assert updated[0]["lease_state"] == "finished"

    def test_no_results_leaves_unchanged(self):
        rows = [
            {"run_id": "run_1", "lease_state": "queued", "ended_at": None, "result_status": None},
        ]
        updated = reconcile_queue_with_results(rows, {})
        assert updated[0]["lease_state"] == "queued"


# ---------------------------------------------------------------------------
# summarize_queue
# ---------------------------------------------------------------------------

class TestSummarizeQueue:
    def test_counts_states(self):
        rows = [
            {"lease_state": "queued", "wave": "w1", "result_status": None},
            {"lease_state": "running", "wave": "w1", "result_status": None},
            {"lease_state": "completed", "wave": "w1", "result_status": "completed"},
            {"lease_state": "finished", "wave": "w1", "result_status": "failed"},
        ]
        summary = summarize_queue(rows)
        assert summary["queue_total"] == 4
        assert summary["queue_queued"] == 1
        assert summary["queue_running"] == 1
        assert summary["queue_finished"] == 2
        assert summary["queue_completed"] == 1

    def test_wave_filter(self):
        rows = [
            {"lease_state": "queued", "wave": "w1"},
            {"lease_state": "queued", "wave": "w2"},
            {"lease_state": "running", "wave": "w1"},
        ]
        summary = summarize_queue(rows, wave_name="w1")
        assert summary["queue_total"] == 2
        assert summary["queue_queued"] == 1
        assert summary["queue_running"] == 1


# ---------------------------------------------------------------------------
# wave_rows and wave_result_rows
# ---------------------------------------------------------------------------

class TestWaveHelpers:
    def test_wave_rows_filters(self):
        rows = [
            {"wave": "w1", "run_id": "r1"},
            {"wave": "w2", "run_id": "r2"},
            {"wave": "w1", "run_id": "r3"},
        ]
        filtered = wave_rows(rows, "w1")
        assert len(filtered) == 2
        assert all(r["wave"] == "w1" for r in filtered)

    def test_wave_result_rows(self):
        queue = [
            {"wave": "w1", "run_id": "r1"},
            {"wave": "w1", "run_id": "r2"},
            {"wave": "w2", "run_id": "r3"},
        ]
        result_index = {
            "r1": {"status": "completed", "val_bpb": 1.2},
            "r3": {"status": "completed", "val_bpb": 1.0},
        }
        results = wave_result_rows(queue, result_index, "w1")
        assert len(results) == 1
        assert results[0]["val_bpb"] == 1.2


# ---------------------------------------------------------------------------
# results_by_id
# ---------------------------------------------------------------------------

class TestResultsById:
    def test_indexes_by_id(self):
        rows = [
            {"id": "a", "val": 1},
            {"id": "b", "val": 2},
        ]
        index = results_by_id(rows)
        assert index["a"]["val"] == 1
        assert index["b"]["val"] == 2

    def test_skips_rows_without_id(self):
        rows = [
            {"id": "a", "val": 1},
            {"val": 2},
        ]
        index = results_by_id(rows)
        assert len(index) == 1
