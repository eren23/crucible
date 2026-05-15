"""Persistent state management for the autonomous research loop.

Tracks hypothesis queue, experiment history, active beliefs, and budget.
Storage format: JSONL with timestamped entries.

Concurrency: multiple autonomous-loop sessions in the same project must
not silently overwrite each other. ``write_lock()`` provides an advisory
exclusive lock on ``{state_file}.lock`` using ``fcntl.flock``, plus a
fresh read from disk so callers see peer writes before mutating.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from crucible.core.errors import StateLockTimeout
from crucible.core.log import utc_now_iso

DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_SUFFIX = ".lock"

_WINDOWS_FALLBACK_WARNED = False


def _warn_windows_fallback_once() -> None:
    global _WINDOWS_FALLBACK_WARNED
    if _WINDOWS_FALLBACK_WARNED:
        return
    _WINDOWS_FALLBACK_WARNED = True
    try:
        from crucible.core.log import log_warn
        log_warn(
            "ResearchState.write_lock: fcntl unavailable on this platform; "
            "concurrent autonomous-loop sessions may corrupt research_state.jsonl."
        )
    except ImportError:
        pass


class ResearchState:
    """Persistent research loop state backed by a JSONL file."""

    def __init__(self, state_file: Path, budget_hours: float = 10.0) -> None:
        self.state_file = Path(state_file)
        self._total_budget_hours = budget_hours
        self.hypotheses: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.beliefs: list[str] = []
        self.findings: list[dict[str, Any]] = []
        self._hours_used: float = 0.0
        self._load()

    # ------------------------------------------------------------------
    # Concurrency safety
    # ------------------------------------------------------------------

    def _lock_path(self) -> Path:
        return self.state_file.with_suffix(self.state_file.suffix + _LOCK_SUFFIX)

    @contextmanager
    def write_lock(
        self,
        *,
        timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        poll_interval: float = 0.1,
    ) -> Iterator["ResearchState"]:
        """Acquire an exclusive advisory lock + reload from disk.

        Use for any read-modify-write sequence on the research state file:

        .. code-block:: python

            with state.write_lock():
                state.add_hypothesis(...)
                state.save()  # MUST be called inside the block

        On entry, the lock is acquired and ``_reload_in_place()`` is called
        so the caller sees the latest peer writes. The caller is responsible
        for calling :meth:`save` before exiting the block — otherwise the
        next :meth:`_reload_in_place` silently discards their writes. If an
        exception is raised inside the block, in-memory mutations are
        intentionally NOT auto-saved (the state may be inconsistent).

        Raises :class:`StateLockTimeout` if the lock cannot be acquired
        within ``timeout`` seconds. POSIX-only — on Windows, falls back to
        a no-op (with a one-time warning) and concurrency safety is lost.

        Do not nest. ``fcntl.flock`` semantics for same-process re-entry
        on a fresh fd are platform-dependent (BSD-derived macOS will block
        on itself until ``timeout``). NFS / network volumes also have
        undefined locking semantics — keep ``state_file`` on local disk.
        """
        try:
            import fcntl
        except ImportError:
            _warn_windows_fallback_once()
            self._reload_in_place()
            yield self
            return

        lock_path = self._lock_path()
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
                        raise StateLockTimeout(
                            f"Could not acquire research-state lock at {lock_path} "
                            f"within {timeout:.1f}s — another Crucible process may "
                            f"be holding it."
                        ) from exc
                    time.sleep(poll_interval)
            self._reload_in_place()
            try:
                yield self
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)

    def _reload_in_place(self) -> None:
        """Drop in-memory state and reload from disk.

        Called inside ``write_lock`` so the caller sees the latest peer
        writes before modifying. Equivalent to constructing a fresh
        ``ResearchState`` against the same file, without changing the
        object identity. ``_total_budget_hours`` is preserved across
        reload — ``_load()`` overwrites it via the ``budget_adjustment``
        ledger entry when present, so the constructor default survives
        only until the first ``save()`` (across any process).
        """
        self.hypotheses = []
        self.history = []
        self.beliefs = []
        self.findings = []
        self._hours_used = 0.0
        self._load()

    def snapshot(self) -> dict[str, Any]:
        """Return an opaque snapshot for stale-submit detection.

        Includes both four list-length counters AND a content hash so the
        guard catches mutations that don't change a list length:
        ``mark_hypothesis`` (status flip in place), ``update_beliefs``
        (wholesale replacement with same length), ``charge_hours``
        (budget-only mutation). The hash is a truncated sha256 of a
        stable JSON serialization of all mutable fields.

        Iteration is intentionally not part of the snapshot — it is
        caller-controlled and would cause spurious mismatches when the
        same state is queried at different loop turns. Loop turn
        identity is tracked separately by the autonomous-loop session
        driver.
        """
        digest_input = json.dumps(
            {
                "hypotheses": [
                    (h.get("hypothesis", h.get("name", "")), h.get("status", "pending"))
                    for h in self.hypotheses
                ],
                "history": [
                    (rec.get("experiment", {}).get("name", ""), rec.get("ts", ""))
                    for rec in self.history
                ],
                "beliefs": list(self.beliefs),
                "findings": [
                    (f.get("finding", ""), f.get("ts", ""), f.get("category", ""))
                    for f in self.findings
                ],
                "hours_used": round(self._hours_used, 6),
            },
            sort_keys=True,
        )
        content_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
        return {
            "history_len": len(self.history),
            "hypotheses_len": len(self.hypotheses),
            "beliefs_len": len(self.beliefs),
            "findings_len": len(self.findings),
            "content_hash": content_hash,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        for line in self.state_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            kind = entry.get("kind")
            if kind == "hypothesis":
                self.hypotheses.append(entry["data"])
            elif kind == "result":
                self.history.append(entry["data"])
                self._hours_used += entry["data"].get("pod_hours", 0.0)
            elif kind == "beliefs":
                self.beliefs = entry["data"]
            elif kind == "finding":
                self.findings.append(entry["data"])
            elif kind == "budget_adjustment":
                self._total_budget_hours = entry["data"]["total_hours"]
                self._hours_used = entry["data"].get("hours_used", self._hours_used)

    def save(self) -> None:
        """Persist full state to JSONL (atomic write)."""
        lines: list[str] = []
        for hyp in self.hypotheses:
            lines.append(json.dumps({"kind": "hypothesis", "ts": hyp.get("ts", utc_now_iso()), "data": hyp}))
        for rec in self.history:
            lines.append(json.dumps({"kind": "result", "ts": rec.get("ts", utc_now_iso()), "data": rec}))
        for finding in self.findings:
            lines.append(json.dumps({"kind": "finding", "ts": finding.get("ts", utc_now_iso()), "data": finding}))
        if self.beliefs:
            lines.append(json.dumps({"kind": "beliefs", "ts": utc_now_iso(), "data": self.beliefs}))
        lines.append(json.dumps({
            "kind": "budget_adjustment",
            "ts": utc_now_iso(),
            "data": {"total_hours": self._total_budget_hours, "hours_used": self._hours_used},
        }))
        payload = "\n".join(lines) + "\n"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.state_file.parent), prefix=self.state_file.name + ".", suffix=".tmp", text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------------
    # Hypothesis management
    # ------------------------------------------------------------------

    def add_hypothesis(self, hypothesis: dict[str, Any]) -> None:
        hypothesis.setdefault("ts", utc_now_iso())
        hypothesis.setdefault("status", "pending")
        self.hypotheses.append(hypothesis)
        self.hypotheses.sort(key=lambda h: -h.get("expected_impact", h.get("expected_bpb_impact", 0.0)))

    def pending_hypotheses(self) -> list[dict[str, Any]]:
        return [h for h in self.hypotheses if h.get("status") == "pending"]

    def mark_hypothesis(self, hypothesis_name: str, status: str) -> None:
        for h in self.hypotheses:
            if h.get("hypothesis", h.get("name", "")) == hypothesis_name:
                h["status"] = status
                break

    # ------------------------------------------------------------------
    # Experiment history
    # ------------------------------------------------------------------

    def record_result(self, experiment: dict[str, Any], result: dict[str, Any]) -> None:
        entry = {
            "ts": utc_now_iso(),
            "experiment": experiment,
            "result": result,
            "pod_hours": experiment.get("pod_hours", 0.0),
        }
        self.history.append(entry)
        self._hours_used += entry["pod_hours"]

    def get_history_summary(self, primary_metric: str = "val_loss") -> str:
        if not self.history:
            return "No experiments completed yet."
        lines = [f"Experiment history ({len(self.history)} runs, {self._hours_used:.2f} compute-hours used):"]
        recent = self.history[-20:]
        for rec in recent:
            exp = rec.get("experiment", {})
            res = rec.get("result", {})
            name = exp.get("name", "unknown")
            # Try to extract the primary metric from result
            metric_val = res.get(primary_metric, res.get("result", {}).get(primary_metric))
            status = res.get("status", "unknown")
            metric_str = f"{metric_val:.4f}" if isinstance(metric_val, (int, float)) else str(metric_val)
            lines.append(f"  {name}: {primary_metric}={metric_str} status={status}")
        if len(self.history) > 20:
            lines.append(f"  ... ({len(self.history) - 20} earlier runs omitted)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Beliefs
    # ------------------------------------------------------------------

    def update_beliefs(self, beliefs: list[str]) -> None:
        self.beliefs = list(beliefs)

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self._total_budget_hours - self._hours_used)

    def charge_hours(self, hours: float) -> None:
        self._hours_used += hours

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def add_finding(
        self,
        finding: str,
        category: str = "observation",
        source_experiments: list[str] | None = None,
        confidence: float = 0.7,
        created_by: str = "unknown",
    ) -> dict[str, Any]:
        """Record a research finding."""
        entry: dict[str, Any] = {
            "ts": utc_now_iso(),
            "finding": finding,
            "category": category,
            "source_experiments": source_experiments or [],
            "confidence": confidence,
            "created_by": created_by,
        }
        self.findings.append(entry)
        return entry

    def get_findings(self, category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return findings, optionally filtered by category."""
        if category:
            filtered = [f for f in self.findings if f.get("category") == category]
        else:
            filtered = list(self.findings)
        return filtered[-limit:]
