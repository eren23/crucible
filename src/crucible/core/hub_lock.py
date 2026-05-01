"""Cross-process file lock for ``~/.crucible-hub/`` registries.

Two Crucible projects on the same machine share ``~/.crucible-hub/``. Any
read-filter-write sequence on a shared file (``installed.yaml``,
``taps.yaml``, ``hub.yaml``, the architecture registry, finding ledgers)
is racy without serialization: the second writer overwrites the first.

This module provides a single ``hub_lock()`` context manager that acquires
an advisory exclusive lock on ``{hub_dir}/.lock``. All callers that mutate
shared state should wrap their critical section in ``with hub_lock(hub):``.
``append_jsonl`` already uses per-file ``fcntl.flock`` for append-only
ledgers; this lock covers the broader read-filter-write pattern that
``write_jsonl`` cannot make atomic on its own.

The lock is process-wide, advisory, and POSIX-only. Other tools that
touch the hub directory without acquiring this lock are not protected —
that's an accepted limitation; we control all writers.
"""
from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_LOCK_FILENAME = ".lock"

_WINDOWS_FALLBACK_WARNED = False


class HubLockTimeout(RuntimeError):
    """Raised when ``hub_lock`` cannot acquire within the timeout."""


def _warn_windows_fallback_once() -> None:
    """Surface the no-op fallback exactly once per process.

    On non-POSIX hosts (Windows without fcntl) ``hub_lock`` degrades to a
    no-op so callers don't crash — but multi-project safety is silently
    lost. A loud one-time warning makes that visible without spamming
    every critical section.
    """
    global _WINDOWS_FALLBACK_WARNED
    if _WINDOWS_FALLBACK_WARNED:
        return
    _WINDOWS_FALLBACK_WARNED = True
    try:
        from crucible.core.log import log_warn
        log_warn(
            "hub_lock: fcntl unavailable on this platform; cross-project "
            "registry safety is degraded. Concurrent Crucible processes "
            "may corrupt ~/.crucible-hub state."
        )
    except ImportError:
        pass  # log module not available (e.g. tests bootstrapping); silent fallback


@contextmanager
def hub_lock(
    hub_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval: float = 0.1,
    lock_filename: str = DEFAULT_LOCK_FILENAME,
) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``{hub_dir}/{lock_filename}``.

    Blocks up to ``timeout`` seconds waiting for the lock. Raises
    :class:`HubLockTimeout` if it cannot acquire within the window. The
    lock is released automatically when the context exits, including on
    exception.

    Uses ``fcntl.flock`` (POSIX advisory locking). If ``fcntl`` is not
    importable (e.g. Windows), falls back to a no-op so callers continue
    to function — but they lose the cross-project safety guarantee. We do
    not target Windows in production.
    """
    try:
        import fcntl
    except ImportError:
        # Non-POSIX platform — give up on locking but don't crash callers.
        _warn_windows_fallback_once()
        yield
        return

    hub_dir.mkdir(parents=True, exist_ok=True)
    lock_path = hub_dir / lock_filename
    deadline = time.monotonic() + timeout

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise HubLockTimeout(
                        f"Could not acquire hub lock at {lock_path} within "
                        f"{timeout:.1f}s — another Crucible process may be "
                        f"holding it."
                    ) from exc
                time.sleep(poll_interval)
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)
