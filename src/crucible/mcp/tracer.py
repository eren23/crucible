"""Session trace recorder for MCP tool calls."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from crucible.core.redact import redact_secrets


class SessionTracer:
    """Records MCP tool calls to a JSONL trace file with redacted secrets."""

    def __init__(self, trace_dir: Path, session_id: str | None = None):
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / f"{self.session_id}.jsonl"
        self.meta_path = self.trace_dir / f"{self.session_id}.meta.yaml"
        self._seq = 0
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._tool_counts: dict[str, int] = {}
        self._identifiers: dict[str, set[str]] = {}

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
        duration_ms: float,
        status: str = "ok",
        error: str | None = None,
        identifiers: dict[str, Any] | None = None,
    ) -> None:
        """Append a redacted tool call to the trace JSONL."""
        self._seq += 1
        self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1
        normalized_ids = _normalize_identifiers(identifiers or {})
        _accumulate_identifiers(self._identifiers, normalized_ids)

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "seq": self._seq,
            "tool": tool,
            "arguments": redact_secrets(arguments),
            "result": redact_secrets(result) if isinstance(result, (dict, list)) else _truncate(str(result), 2000),
            "duration_ms": round(duration_ms, 1),
            "status": status,
        }
        if error:
            entry["error"] = error
        if normalized_ids:
            entry["identifiers"] = normalized_ids

        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def finalize(self) -> Path:
        """Write the .meta.yaml summary and return the trace path."""
        meta = {
            "session_id": self.session_id,
            "trace_version": 2,
            "started_at": self._started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "tool_calls": self._seq,
            "tool_counts": dict(sorted(self._tool_counts.items(), key=lambda x: -x[1])),
            "trace_file": self.trace_path.name,
        }
        if self._identifiers:
            meta["identifiers"] = {
                key: sorted(values)
                for key, values in sorted(self._identifiers.items())
                if values
            }
        # Atomic write
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(yaml.dump(meta, default_flow_style=False, sort_keys=False), encoding="utf-8")
        tmp.rename(self.meta_path)
        return self.trace_path


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... ({len(s) - max_len} chars truncated)"


def _normalize_identifiers(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            cleaned = [str(v) for v in value if v not in (None, "")]
            if cleaned:
                normalized[key] = cleaned
        else:
            normalized[key] = str(value)
    return normalized


def _accumulate_identifiers(target: dict[str, set[str]], values: dict[str, Any]) -> None:
    for key, value in values.items():
        bucket = target.setdefault(key, set())
        if isinstance(value, list):
            bucket.update(str(v) for v in value)
        else:
            bucket.add(str(value))


def load_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Load a trace JSONL file into a list of entries."""
    entries = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_trace_meta(meta_path: Path) -> dict[str, Any]:
    """Load a trace metadata YAML file."""
    return yaml.safe_load(meta_path.read_text(encoding="utf-8"))
