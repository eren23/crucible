"""Tests for crucible.researcher.state."""
from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import pytest

from crucible.core.errors import StateLockTimeout
from crucible.researcher.state import ResearchState


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestEmptyState:
    def test_budget_remaining(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.budget_remaining == 5.0

    def test_empty_hypotheses(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.hypotheses == []

    def test_empty_history(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.history == []

    def test_empty_beliefs(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=5.0)
        assert state.beliefs == []


# ---------------------------------------------------------------------------
# Hypothesis management
# ---------------------------------------------------------------------------

class TestHypothesisManagement:
    def test_add_hypothesis(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "test", "expected_impact": 0.5, "config": {"A": "1"}})
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0]["name"] == "test"
        assert state.hypotheses[0]["status"] == "pending"

    def test_hypotheses_sorted_by_impact(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "low", "expected_impact": 0.1, "config": {}})
        state.add_hypothesis({"name": "high", "expected_impact": 0.9, "config": {}})
        state.add_hypothesis({"name": "mid", "expected_impact": 0.5, "config": {}})
        # Highest impact first
        assert state.hypotheses[0]["name"] == "high"
        assert state.hypotheses[1]["name"] == "mid"
        assert state.hypotheses[2]["name"] == "low"

    def test_pending_hypotheses(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "a", "expected_impact": 0.5, "config": {}})
        state.add_hypothesis({"name": "b", "expected_impact": 0.3, "config": {}})
        state.mark_hypothesis("a", "tested")
        pending = state.pending_hypotheses()
        assert len(pending) == 1
        assert pending[0]["name"] == "b"

    def test_mark_hypothesis(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"hypothesis": "h1", "expected_impact": 0.5, "config": {}})
        state.mark_hypothesis("h1", "tested")
        assert state.hypotheses[0]["status"] == "tested"

    def test_add_hypothesis_sets_timestamp(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "ts_test", "config": {}})
        assert "ts" in state.hypotheses[0]

    def test_hypotheses_sorted_by_expected_bpb_impact(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "a", "expected_bpb_impact": 0.2, "config": {}})
        state.add_hypothesis({"name": "b", "expected_bpb_impact": 0.8, "config": {}})
        assert state.hypotheses[0]["name"] == "b"


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

class TestBudgetTracking:
    def test_record_result_charges_budget(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 2.0},
            result={"status": "completed", "val_bpb": 1.2},
        )
        assert state.budget_remaining == 8.0

    def test_multiple_charges(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 3.0},
            result={"status": "completed"},
        )
        state.record_result(
            experiment={"name": "exp2", "pod_hours": 4.0},
            result={"status": "completed"},
        )
        assert state.budget_remaining == 3.0

    def test_budget_cannot_go_negative(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=1.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 5.0},
            result={"status": "completed"},
        )
        assert state.budget_remaining == 0.0

    def test_charge_hours(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.charge_hours(2.5)
        assert state.budget_remaining == 7.5


# ---------------------------------------------------------------------------
# Beliefs
# ---------------------------------------------------------------------------

class TestBeliefs:
    def test_update_beliefs(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.update_beliefs(["belief A", "belief B"])
        assert state.beliefs == ["belief A", "belief B"]

    def test_update_beliefs_replaces(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.update_beliefs(["old"])
        state.update_beliefs(["new1", "new2"])
        assert state.beliefs == ["new1", "new2"]


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_record_result_appends_history(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 1.0},
            result={"status": "completed", "val_bpb": 1.5},
        )
        assert len(state.history) == 1
        assert state.history[0]["experiment"]["name"] == "exp1"
        assert state.history[0]["result"]["val_bpb"] == 1.5

    def test_get_history_summary_empty(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        summary = state.get_history_summary()
        assert "No experiments completed yet" in summary

    def test_get_history_summary_with_data(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 1.0},
            result={"status": "completed", "val_loss": 1.2},
        )
        summary = state.get_history_summary()
        assert "exp1" in summary
        assert "1.2" in summary


# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------

class TestSaveAndReload:
    def test_full_save_reload_cycle(self, tmp_path):
        path = tmp_path / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.add_hypothesis({"name": "h1", "expected_impact": 0.3, "config": {"X": "1"}})
        state.update_beliefs(["belief 1", "belief 2"])
        state.record_result(
            experiment={"name": "exp1", "pod_hours": 1.0},
            result={"status": "completed"},
        )
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        assert len(reloaded.hypotheses) == 1
        assert len(reloaded.history) == 1
        assert reloaded.beliefs == ["belief 1", "belief 2"]
        assert reloaded.budget_remaining == 9.0

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.save()
        assert path.exists()

    def test_reload_preserves_hypothesis_order(self, tmp_path):
        path = tmp_path / "state.jsonl"
        state = ResearchState(path, budget_hours=10.0)
        state.add_hypothesis({"name": "low", "expected_impact": 0.1, "config": {}})
        state.add_hypothesis({"name": "high", "expected_impact": 0.9, "config": {}})
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        # After reload, hypotheses are loaded in file order (not re-sorted)
        # The file stores them in the sorted order from add_hypothesis
        assert reloaded.hypotheses[0]["name"] == "high"

    def test_reload_budget_adjustment(self, tmp_path):
        path = tmp_path / "state.jsonl"
        state = ResearchState(path, budget_hours=20.0)
        state.charge_hours(5.0)
        state.save()

        reloaded = ResearchState(path, budget_hours=10.0)
        # Budget adjustment from file overrides constructor param
        assert reloaded.budget_remaining == 15.0


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_empty_snapshot(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        snap = state.snapshot()
        assert snap == {
            "history_len": 0,
            "hypotheses_len": 0,
            "beliefs_len": 0,
            "findings_len": 0,
        }

    def test_snapshot_tracks_mutations(self, tmp_path):
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        state.add_hypothesis({"name": "h1", "config": {}})
        state.update_beliefs(["b1", "b2"])
        state.add_finding("f1", confidence=0.7)
        snap = state.snapshot()
        assert snap == {
            "history_len": 0,
            "hypotheses_len": 1,
            "beliefs_len": 2,
            "findings_len": 1,
        }

    def test_snapshot_does_not_include_iteration(self, tmp_path):
        """Iteration is loop-turn identity, not state — tracked separately."""
        state = ResearchState(tmp_path / "state.jsonl", budget_hours=10.0)
        snap = state.snapshot()
        assert "iteration" not in snap


# ---------------------------------------------------------------------------
# Concurrency — write_lock
# ---------------------------------------------------------------------------

def _holder_process(state_path_str: str, hold_seconds: float, ready_marker_str: str) -> None:
    """Subprocess target: acquire the lock and hold it for *hold_seconds*."""
    from pathlib import Path as _Path
    from crucible.researcher.state import ResearchState as _State

    state = _State(_Path(state_path_str), budget_hours=10.0)
    with state.write_lock(timeout=5.0):
        _Path(ready_marker_str).write_text("ready", encoding="utf-8")
        time.sleep(hold_seconds)


def _writer_process(state_path_str: str, name: str) -> None:
    """Subprocess target: lock, add a hypothesis, save, release."""
    from pathlib import Path as _Path
    from crucible.researcher.state import ResearchState as _State

    state = _State(_Path(state_path_str), budget_hours=10.0)
    with state.write_lock(timeout=10.0):
        state.add_hypothesis({"name": name, "expected_impact": 0.5, "config": {}})
        state.save()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl locks are POSIX-only")
class TestWriteLockConcurrency:
    def test_write_lock_blocks_concurrent_acquire(self, tmp_path):
        state_path = tmp_path / "state.jsonl"
        ready_marker = tmp_path / "ready"
        # Initialise the file so the holder doesn't race on _load().
        ResearchState(state_path, budget_hours=10.0).save()

        ctx = multiprocessing.get_context("spawn")
        holder = ctx.Process(
            target=_holder_process,
            args=(str(state_path), 3.0, str(ready_marker)),
            daemon=True,
        )
        holder.start()
        try:
            deadline = time.monotonic() + 5.0
            while not ready_marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready_marker.exists(), "holder failed to acquire lock"

            state = ResearchState(state_path, budget_hours=10.0)
            with pytest.raises(StateLockTimeout):
                with state.write_lock(timeout=0.5):
                    pass
        finally:
            holder.join(timeout=10.0)
            if holder.is_alive():
                holder.terminate()

    def test_write_lock_serializes_writers(self, tmp_path):
        state_path = tmp_path / "state.jsonl"
        ResearchState(state_path, budget_hours=10.0).save()

        ctx = multiprocessing.get_context("spawn")
        writers = [
            ctx.Process(target=_writer_process, args=(str(state_path), f"hyp_{i}"))
            for i in range(5)
        ]
        for w in writers:
            w.start()

        # Single deadline for all writers: 5 processes × (~3s spawn + 1s save)
        # is the worst case on macOS spawn-mode. Use a shared deadline rather
        # than per-process so a slow first process doesn't starve the last.
        deadline = time.monotonic() + 60.0
        for w in writers:
            remaining = max(0.5, deadline - time.monotonic())
            w.join(timeout=remaining)
            assert w.exitcode == 0, f"writer pid={w.pid} exitcode={w.exitcode}"

        # All five writes should be present — no last-write-wins.
        final = ResearchState(state_path, budget_hours=10.0)
        names = sorted(h.get("name", "") for h in final.hypotheses)
        assert names == [f"hyp_{i}" for i in range(5)]

    def test_write_lock_without_save_loses_writes(self, tmp_path):
        """Document the save-or-lose contract: forgetting save() loses writes."""
        state_path = tmp_path / "state.jsonl"
        state = ResearchState(state_path, budget_hours=10.0)
        state.add_hypothesis({"name": "persisted", "config": {}})
        state.save()

        with state.write_lock():
            state.add_hypothesis({"name": "forgotten", "config": {}})
            # Caller intentionally forgets state.save() — simulates a buggy
            # session driver. The lock releases cleanly on exit.

        # Next acquisition reloads from disk; the forgotten write is lost.
        with state.write_lock():
            names = {h.get("name") for h in state.hypotheses}
            assert names == {"persisted"}, f"forgotten write leaked: {names}"

    def test_write_lock_reloads_from_disk(self, tmp_path):
        state_path = tmp_path / "state.jsonl"
        # Parent constructs state and saves an initial hypothesis.
        parent = ResearchState(state_path, budget_hours=10.0)
        parent.add_hypothesis({"name": "parent_h", "config": {}})
        parent.save()

        # Peer process writes a new hypothesis to disk.
        ctx = multiprocessing.get_context("spawn")
        peer = ctx.Process(target=_writer_process, args=(str(state_path), "peer_h"))
        peer.start()
        peer.join(timeout=10.0)
        assert peer.exitcode == 0

        # Parent's in-memory state is stale (still 1). write_lock must reload
        # so the parent sees both hypotheses before mutating.
        assert len(parent.hypotheses) == 1
        with parent.write_lock():
            assert len(parent.hypotheses) == 2
            names = {h.get("name") for h in parent.hypotheses}
            assert names == {"parent_h", "peer_h"}
