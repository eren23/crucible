"""Persistent state management for the autonomous research loop.

Tracks hypothesis queue, experiment history, active beliefs, and budget.
Storage format: JSONL with timestamped entries.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from crucible.core.log import utc_now_iso


class ResearchState:
    """Persistent research loop state backed by a JSONL file."""

    def __init__(self, state_file: Path, budget_hours: float = 10.0) -> None:
        self.state_file = Path(state_file)
        self._total_budget_hours = budget_hours
        self.hypotheses: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.beliefs: list[str] = []
        self._hours_used: float = 0.0
        self._load()

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
