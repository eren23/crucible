"""Tests for crucible.runner.tracker."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from crucible.runner.tracker import RunTracker


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

class TestRunTrackerCreation:
    def test_creates_status_file(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        assert tracker.status_path.exists()

    def test_creates_log_path(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        assert tracker.log_path == tmp_path / "logs" / "test_run.txt"

    def test_initial_state_is_created(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        assert tracker.state == "created"

    def test_status_contains_run_id(self, tmp_path):
        tracker = RunTracker("my_exp", out_dir=tmp_path / "logs", project_root=tmp_path)
        assert tracker.status["run_id"] == "my_exp"

    def test_status_contains_host_and_pid(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        assert "host" in tracker.status
        assert "pid" in tracker.status

    def test_creates_output_directory(self, tmp_path):
        out_dir = tmp_path / "deep" / "logs"
        tracker = RunTracker("test_run", out_dir=out_dir, project_root=tmp_path)
        assert out_dir.exists()

    def test_loads_existing_status(self, tmp_path):
        out_dir = tmp_path / "logs"
        tracker1 = RunTracker("resume_run", out_dir=out_dir, project_root=tmp_path)
        tracker1.update(state="running", phase="training")

        tracker2 = RunTracker("resume_run", out_dir=out_dir, project_root=tmp_path)
        assert tracker2.state == "running"


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestRunTrackerUpdate:
    def test_update_changes_state(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.update(state="running")
        assert tracker.state == "running"

    def test_update_adds_custom_fields(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.update(phase="training", step=10)
        status = tracker.status
        assert status["phase"] == "training"
        assert status["step"] == 10

    def test_update_writes_to_disk(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.update(state="running")
        data = json.loads(tracker.status_path.read_text(encoding="utf-8"))
        assert data["state"] == "running"

    def test_update_bumps_heartbeat(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        initial = tracker.status.get("last_heartbeat_at")
        tracker.update(state="running", heartbeat=True)
        assert tracker.status["last_heartbeat_at"] is not None

    def test_update_no_heartbeat(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.update(state="running", heartbeat=False)
        assert "updated_at" in tracker.status


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestRunTrackerHeartbeat:
    def test_heartbeat_updates_phase(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.heartbeat("training", step=42, total_steps=100)
        status = tracker.status
        assert status["phase"] == "training"
        assert status["step"] == 42

    def test_heartbeat_bumps_timestamp(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.heartbeat("launching")
        assert "last_heartbeat_at" in tracker.status


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

class TestRunTrackerFinalize:
    def test_finalize_sets_state(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.finalize("completed", val_bpb=1.23)
        assert tracker.state == "completed"

    def test_finalize_sets_ended_at(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.finalize("failed", error="OOM")
        assert "ended_at" in tracker.status

    def test_finalize_includes_extra_fields(self, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.finalize("completed", val_bpb=1.1, model_bytes=50000)
        status = tracker.status
        assert status["val_bpb"] == 1.1
        assert status["model_bytes"] == 50000


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestRunTrackerWriteManifest:
    @patch("crucible.runner.tracker.code_fingerprint", return_value={"fingerprint": "abc", "files": {}})
    @patch("crucible.runner.tracker.safe_git_sha", return_value="deadbeef")
    @patch("crucible.runner.tracker.safe_git_dirty", return_value=False)
    def test_writes_manifest_file(self, mock_dirty, mock_sha, mock_fp, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.write_manifest(
            backend="torch",
            script_path="train.py",
            config={"LR": "0.001"},
            tags=["test"],
        )
        assert tracker.manifest_path.exists()
        manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == "test_run"
        assert manifest["backend"] == "torch"
        assert manifest["config"]["LR"] == "0.001"
        assert manifest["tags"] == ["test"]
        assert manifest["git_sha"] == "deadbeef"
        assert manifest["git_dirty"] is False

    @patch("crucible.runner.tracker.code_fingerprint", return_value={"fingerprint": "abc", "files": {}})
    @patch("crucible.runner.tracker.safe_git_sha", return_value=None)
    @patch("crucible.runner.tracker.safe_git_dirty", return_value=None)
    def test_manifest_handles_no_git(self, mock_dirty, mock_sha, mock_fp, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.write_manifest(
            backend="mlx",
            script_path="train_mlx.py",
            config={},
        )
        manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
        assert manifest["git_sha"] is None
        assert manifest["git_dirty"] is None

    @patch("crucible.runner.tracker.code_fingerprint", return_value={"fingerprint": "abc", "files": {}})
    @patch("crucible.runner.tracker.safe_git_sha", return_value="abc123")
    @patch("crucible.runner.tracker.safe_git_dirty", return_value=False)
    def test_manifest_merges_on_rewrite(self, mock_dirty, mock_sha, mock_fp, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.write_manifest(
            backend="torch",
            script_path="train.py",
            config={"LR": "0.001"},
            tags=["first"],
            extra={"custom_field": "v1"},
        )
        # Re-write (e.g., after OOM retry)
        tracker.write_manifest(
            backend="torch",
            script_path="train.py",
            config={"LR": "0.002"},
            tags=["second"],
        )
        manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
        # Should have merged: new values overwrite
        assert manifest["config"]["LR"] == "0.002"
        assert manifest["tags"] == ["second"]
        # The custom_field from first write is preserved
        assert manifest["custom_field"] == "v1"

    @patch("crucible.runner.tracker.code_fingerprint", return_value={"fingerprint": "abc", "files": {}})
    @patch("crucible.runner.tracker.safe_git_sha", return_value=None)
    @patch("crucible.runner.tracker.safe_git_dirty", return_value=None)
    def test_manifest_extra_fields(self, mock_dirty, mock_sha, mock_fp, tmp_path):
        tracker = RunTracker("test_run", out_dir=tmp_path / "logs", project_root=tmp_path)
        tracker.write_manifest(
            backend="torch",
            script_path="train.py",
            config={},
            extra={"parent_run_id": "parent_123", "wave": "auto_iter_1"},
        )
        manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
        assert manifest["parent_run_id"] == "parent_123"
        assert manifest["wave"] == "auto_iter_1"
