"""Tests for crucible.core.file_lock — the shared fcntl primitive."""
from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import pytest

from crucible.core.errors import CrucibleError
from crucible.core.file_lock import FileLockTimeout, file_lock


class _CustomTimeout(CrucibleError):
    pass


def _hold_lock_worker(lock_path_str: str, hold_seconds: float, ready_marker_str: str) -> None:
    """Subprocess target: acquire the lock and hold it."""
    from pathlib import Path as _Path
    from crucible.core.file_lock import file_lock as _fl

    lock_path = _Path(lock_path_str)
    with _fl(lock_path, timeout=5.0):
        _Path(ready_marker_str).write_text("ready", encoding="utf-8")
        time.sleep(hold_seconds)


class TestFileLockDefault:
    def test_acquires_and_releases(self, tmp_path: Path):
        lock = tmp_path / "test.lock"
        with file_lock(lock, timeout=1.0):
            assert lock.exists()
        # Can re-acquire after release
        with file_lock(lock, timeout=1.0):
            pass

    def test_default_timeout_raises_FileLockTimeout(self, tmp_path: Path):
        """Without on_timeout factory, default FileLockTimeout is raised."""
        if sys.platform == "win32":
            pytest.skip("fcntl locks are POSIX-only")

        lock = tmp_path / "default.lock"
        ready = tmp_path / "ready"
        ctx = multiprocessing.get_context("spawn")
        holder = ctx.Process(
            target=_hold_lock_worker, args=(str(lock), 3.0, str(ready)), daemon=True
        )
        holder.start()
        try:
            deadline = time.monotonic() + 5.0
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready.exists(), "holder did not acquire"

            with pytest.raises(FileLockTimeout):
                with file_lock(lock, timeout=0.5):
                    pass
        finally:
            holder.join(timeout=10.0)
            if holder.is_alive():
                holder.terminate()


class TestFileLockOnTimeoutFactory:
    def test_custom_factory_raises_specified_exception(self, tmp_path: Path):
        if sys.platform == "win32":
            pytest.skip("fcntl locks are POSIX-only")

        lock = tmp_path / "custom.lock"
        ready = tmp_path / "ready"
        ctx = multiprocessing.get_context("spawn")
        holder = ctx.Process(
            target=_hold_lock_worker, args=(str(lock), 3.0, str(ready)), daemon=True
        )
        holder.start()
        try:
            deadline = time.monotonic() + 5.0
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready.exists()

            with pytest.raises(_CustomTimeout, match="my-resource"):
                with file_lock(
                    lock,
                    timeout=0.5,
                    on_timeout=lambda msg: _CustomTimeout(
                        f"my-resource: {msg}"
                    ),
                ):
                    pass
        finally:
            holder.join(timeout=10.0)
            if holder.is_alive():
                holder.terminate()


class TestFileLockReleasesOnException:
    def test_lock_released_when_block_raises(self, tmp_path: Path):
        """If the critical section raises, the lock is still released so
        a subsequent acquire succeeds immediately."""
        lock = tmp_path / "release.lock"
        with pytest.raises(RuntimeError, match="boom"):
            with file_lock(lock, timeout=1.0):
                raise RuntimeError("boom")
        # Immediate re-acquire should work.
        with file_lock(lock, timeout=0.1):
            pass


class TestFileLockMkdir:
    def test_creates_parent_dir(self, tmp_path: Path):
        """Lock under a nonexistent parent should still work."""
        lock = tmp_path / "nested" / "deep" / "test.lock"
        with file_lock(lock, timeout=1.0):
            assert lock.exists()
            assert lock.parent.is_dir()
