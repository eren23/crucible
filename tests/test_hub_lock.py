"""Cross-process file lock for ``~/.crucible-hub/`` registries.

Verifies that ``hub_lock`` serializes critical sections in the same
process and times out when another holder has the lock. Cross-process
mutual exclusion is checked by spawning a child process that holds the
lock and confirming the parent times out.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from crucible.core.hub_lock import HubLockTimeout, hub_lock


def test_hub_lock_serializes_within_process(tmp_path: Path):
    """Two ``with hub_lock(...)`` blocks in the same thread don't deadlock."""
    with hub_lock(tmp_path):
        pass
    with hub_lock(tmp_path):
        pass


def test_hub_lock_creates_lock_file(tmp_path: Path):
    with hub_lock(tmp_path):
        assert (tmp_path / ".lock").exists()


def _hold_lock(hub_dir_str: str, hold_seconds: float, ready_path: str) -> None:
    """Helper for cross-process tests: acquire the lock and block."""
    from crucible.core.hub_lock import hub_lock as inner

    hub_dir = Path(hub_dir_str)
    with inner(hub_dir):
        Path(ready_path).write_text("ready", encoding="utf-8")
        time.sleep(hold_seconds)


def test_hub_lock_blocks_when_held_by_another_process(tmp_path: Path):
    """Another process holding the lock must cause the parent to time out."""
    ready_marker = tmp_path / "child_ready"
    proc = mp.Process(
        target=_hold_lock,
        args=(str(tmp_path), 1.5, str(ready_marker)),
    )
    proc.start()
    try:
        # Wait for the child to acquire and signal readiness, with a
        # generous fail-safe so a hang here doesn't wedge the suite.
        deadline = time.monotonic() + 5.0
        while not ready_marker.exists():
            if time.monotonic() >= deadline:
                pytest.fail("child process never acquired hub_lock")
            time.sleep(0.05)

        start = time.monotonic()
        with pytest.raises(HubLockTimeout):
            with hub_lock(tmp_path, timeout=0.3, poll_interval=0.05):
                pytest.fail("parent should not have acquired the lock")
        elapsed = time.monotonic() - start
        # Generous bound — slow CI may add overhead.
        assert 0.25 <= elapsed <= 2.0
    finally:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)


def test_hub_lock_releases_on_exception(tmp_path: Path):
    """An exception inside the ``with`` block must still release the lock."""
    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with hub_lock(tmp_path, timeout=0.5):
            raise _Boom("inside critical section")

    # Should be acquirable again immediately.
    with hub_lock(tmp_path, timeout=0.5):
        pass
