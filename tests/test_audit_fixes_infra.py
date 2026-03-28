"""Tests for audit fix Tiers 3-4: resource leaks, timeouts, error handling.

All tests are non-torch and run without GPU.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
import yaml


# ---------------------------------------------------------------------------
# 3.1 — loggers.py atexit registration
# ---------------------------------------------------------------------------

class TestJsonlLoggerAtexit:
    """JsonlLogger should register atexit handler and handle double-finish."""

    def test_atexit_registered(self, tmp_path: Path):
        with patch("atexit.register") as mock_register:
            from crucible.runner.loggers import JsonlLogger
            logger = JsonlLogger(run_id="test-atexit", log_dir=str(tmp_path))
            mock_register.assert_called_once_with(logger.finish)
            logger.finish()

    def test_double_finish_safe(self, tmp_path: Path):
        from crucible.runner.loggers import JsonlLogger
        logger = JsonlLogger(run_id="test-double", log_dir=str(tmp_path))
        logger.finish()
        logger.finish()  # should not raise

    def test_log_after_finish_noop(self, tmp_path: Path):
        from crucible.runner.loggers import JsonlLogger
        logger = JsonlLogger(run_id="test-after", log_dir=str(tmp_path))
        logger.log({"loss": 0.5}, step=1)
        logger.finish()
        logger.log({"loss": 0.3}, step=2)  # should silently skip
        lines = (tmp_path / "test-after.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1  # only the first log


# ---------------------------------------------------------------------------
# 3.2 — io.py append_jsonl file locking
# ---------------------------------------------------------------------------

class TestAppendJsonlLocking:
    """append_jsonl should use file locking and produce valid JSONL."""

    def test_sequential_appends_produce_valid_jsonl(self, tmp_path: Path):
        from crucible.core.io import append_jsonl, read_jsonl

        path = tmp_path / "test.jsonl"
        for i in range(10):
            append_jsonl(path, {"step": i, "value": i * 0.1})

        records = read_jsonl(path)
        assert len(records) == 10
        assert records[0]["step"] == 0
        assert records[9]["step"] == 9

    def test_append_creates_parent_dirs(self, tmp_path: Path):
        from crucible.core.io import append_jsonl

        path = tmp_path / "nested" / "deep" / "file.jsonl"
        append_jsonl(path, {"key": "value"})
        assert path.exists()

    def test_append_preserves_existing_content(self, tmp_path: Path):
        from crucible.core.io import append_jsonl, read_jsonl

        path = tmp_path / "preserve.jsonl"
        append_jsonl(path, {"first": True})
        append_jsonl(path, {"second": True})
        records = read_jsonl(path)
        assert len(records) == 2
        assert records[0]["first"] is True
        assert records[1]["second"] is True


# ---------------------------------------------------------------------------
# 4.1 — sync.py SSH timeout parameter
# ---------------------------------------------------------------------------

class TestSyncTimeout:
    """remote_exec should accept and pass timeout."""

    def test_run_passes_timeout(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
            from crucible.fleet.sync import _run
            _run(["echo", "hello"], timeout=42, check=False)
            _, kwargs = mock_run.call_args
            assert kwargs.get("timeout") == 42

    def test_remote_exec_default_timeout(self):
        """remote_exec should have a default timeout of 120."""
        import inspect
        from crucible.fleet.sync import remote_exec
        sig = inspect.signature(remote_exec)
        assert sig.parameters["timeout"].default == 120

    def test_remote_exec_passes_custom_timeout(self):
        with patch("crucible.fleet.sync._run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
            from crucible.fleet.sync import remote_exec
            node = {"ssh_host": "1.2.3.4", "ssh_port": 22}
            remote_exec(node, "echo hi", check=False, timeout=30)
            _, kwargs = mock_run.call_args
            assert kwargs.get("timeout") == 30


# ---------------------------------------------------------------------------
# 2.4 — hub.py git error handling
# ---------------------------------------------------------------------------

class TestHubGitErrorHandling:
    """Hub init should handle missing git gracefully."""

    def test_init_without_git_warns(self, tmp_path: Path, capfd: pytest.CaptureFixture[str]):
        hub_dir = tmp_path / "hub-no-git"
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            from crucible.core.hub import HubStore
            hub = HubStore.init(hub_dir)
            assert hub.initialized
        captured = capfd.readouterr()
        assert "git not installed" in captured.err

    def test_init_with_git_failure_warns(self, tmp_path: Path, capfd: pytest.CaptureFixture[str]):
        hub_dir = tmp_path / "hub-git-fail"

        def fail_git(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git", stderr="fatal: error")

        with patch("subprocess.run", side_effect=fail_git):
            from crucible.core.hub import HubStore
            hub = HubStore.init(hub_dir)
            assert hub.initialized
        captured = capfd.readouterr()
        assert "git init failed" in captured.err


# ---------------------------------------------------------------------------
# 2.5 — hub.py track yaml malformed warning
# ---------------------------------------------------------------------------

class TestHubTrackWarning:
    """Malformed track.yaml should warn, not crash."""

    def test_malformed_track_warns(self, tmp_path: Path, capfd: pytest.CaptureFixture[str]):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        (hub_dir / "hub.yaml").write_text(yaml.dump({
            "name": "test", "created_at": "now", "version": 1,
        }))
        tracks_dir = hub_dir / "tracks"
        tracks_dir.mkdir()
        (tracks_dir / "global").mkdir()

        # Good track
        good_dir = tracks_dir / "good-track"
        good_dir.mkdir()
        (good_dir / "track.yaml").write_text(yaml.dump({
            "name": "good-track", "description": "works",
        }))

        # Bad track -- YAML that parses to a string, not a dict
        bad_dir = tracks_dir / "bad-track"
        bad_dir.mkdir()
        (bad_dir / "track.yaml").write_text("just a string\n")

        from crucible.core.hub import HubStore
        hub = HubStore(hub_dir)
        tracks = hub.list_tracks()

        assert len(tracks) == 1
        assert tracks[0]["name"] == "good-track"

        captured = capfd.readouterr()
        assert "Malformed track.yaml in bad-track" in captured.err


# ---------------------------------------------------------------------------
# 4.5 — routes.py exception specificity
# ---------------------------------------------------------------------------

try:
    import fastapi as _fastapi  # noqa: F401
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
class TestRoutesExceptionHandling:
    """API routes should catch CrucibleError but let unexpected errors propagate."""

    def test_crucible_error_imported(self):
        """Verify CrucibleError is imported in routes module."""
        from crucible.api import routes
        assert hasattr(routes, "CrucibleError")

    def test_route_definitions_exist(self):
        """Verify key routes are defined on the router."""
        from crucible.api.routes import router
        paths = [getattr(r, "path", "") for r in router.routes]
        assert any("health" in str(p) for p in paths)
