"""Cross-process advisory file lock via ``fcntl.flock`` (POSIX-only).

The same fcntl.flock + poll-deadline pattern previously lived in five
places: ``hub_lock.py``, ``researcher/state.py::write_lock``,
``researcher/search_tree.py::write_lock``,
``researcher/autonomous_session.py::_file_lock``, and
``researcher/tree_autonomous_session.py::_file_lock``. Two reviewers
flagged this duplication as drift risk — Codex specifically called it
out before Phase 1.5 would add a sixth copy.

This module extracts the common primitive. Each caller passes its
preferred timeout exception via ``on_timeout`` so the type system at
the boundary stays narrow (``StateLockTimeout`` for state, etc.) while
the implementation lives in one place.

Usage::

    from crucible.core.file_lock import file_lock
    from crucible.core.errors import StateLockTimeout

    with file_lock(
        lock_path,
        timeout=30.0,
        on_timeout=lambda msg: StateLockTimeout(msg),
    ):
        ... critical section ...

Higher-level wrappers (``ResearchState.write_lock``,
``SearchTree.write_lock``, the session-driver ``_file_lock`` helpers)
build on this with reload-from-disk-on-entry semantics. They stay in
their own modules; only the raw lock-acquisition primitive is shared.

POSIX-only. Windows degrades to a no-op with a one-time warning so
callers continue to function but lose cross-process safety.
"""
from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from crucible.core.errors import CrucibleError
from crucible.core.log import log_warn

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.1

_WINDOWS_FALLBACK_WARNED = False


class FileLockFactoryError(CrucibleError):
    """``on_timeout`` factory crashed when called with the timeout message.

    Raised when a caller passes a malformed factory (e.g., wrong arity,
    missing required positional arg). The wrapper around
    ``on_timeout(msg)`` catches the inner exception and re-raises this
    with the lock path + cause attached, so an operator gets ``"on_timeout
    factory misconfigured for /path/.lock"`` instead of an opaque
    TypeError thrown from inside fcntl-acquisition.
    """


class FileLockTimeout(CrucibleError):
    """Generic file-lock timeout.

    Callers that want a domain-specific exception type (e.g.,
    ``StateLockTimeout``, ``SearchTreeError``) pass an ``on_timeout``
    factory to :func:`file_lock`. Without one, the timeout path raises
    this default.
    """


def _warn_windows_fallback_once() -> None:
    """Emit a one-time warning that fcntl is unavailable.

    Mirrors :func:`crucible.core.hub_lock._warn_windows_fallback_once`
    and the now-removed ``_warn_tree_windows_fallback_once`` — keeps a
    consistent operator-visible signal that cross-process locking is
    silently off on this platform.
    """
    global _WINDOWS_FALLBACK_WARNED
    if _WINDOWS_FALLBACK_WARNED:
        return
    _WINDOWS_FALLBACK_WARNED = True
    log_warn(
        "file_lock: fcntl unavailable on this platform; cross-process "
        "advisory locking is disabled. Concurrent Crucible processes may "
        "race on shared state."
    )


@contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    on_timeout: Callable[[str], Exception] | None = None,
) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``lock_path`` via ``fcntl.flock``.

    Blocks (with polling) up to ``timeout`` seconds. On timeout, raises
    ``on_timeout(message)`` if a factory is supplied, otherwise
    :class:`FileLockTimeout`. The lock is released automatically when
    the context exits, including on exception inside the block.

    The lockfile is created if it doesn't exist (mode 0o644) and its
    parent directory is mkdir'd. The fd is closed in a ``finally`` so
    crashes inside the block don't leak descriptors.

    Do not nest on the same lock_path from the same process — BSD-style
    ``flock`` semantics on a fresh fd will block on the same process's
    own lock until ``timeout``. (POSIX-style ``flock`` per-open-file-
    description is the same.) Higher-level wrappers should not call
    each other recursively on the same lock.
    """
    try:
        import fcntl
    except ImportError:
        _warn_windows_fallback_once()
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
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
                    msg = (
                        f"Could not acquire file lock at {lock_path} within "
                        f"{timeout:.1f}s — another Crucible process may be holding it."
                    )
                    if on_timeout is not None:
                        try:
                            produced = on_timeout(msg)
                        except BaseException as factory_exc:
                            # Factory itself crashed (e.g., misconfigured
                            # signature). Surface a typed
                            # FileLockFactoryError with the lock path so
                            # debugging doesn't require tracing into
                            # fcntl internals.
                            raise FileLockFactoryError(
                                f"on_timeout factory raised "
                                f"{type(factory_exc).__name__}: {factory_exc}. "
                                f"Lock path: {lock_path}."
                            ) from factory_exc
                        if not isinstance(produced, BaseException):
                            # Factory returned a non-Exception (string,
                            # None, etc.). Same diagnostic shape.
                            raise FileLockFactoryError(
                                f"on_timeout factory returned "
                                f"{type(produced).__name__!r}, expected an "
                                f"exception instance. Lock path: {lock_path}."
                            ) from exc
                        raise produced from exc
                    raise FileLockTimeout(msg) from exc
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


__all__ = [
    "file_lock",
    "FileLockTimeout",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
]
