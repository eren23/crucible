"""Local state tracking for Research DAG ↔ Spider Chat mapping.

Maintains a JSONL ledger mapping Crucible node IDs to Spider Chat canvas node IDs.
This allows bidirectional sync without either system knowing about the other's internals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.io import append_jsonl, atomic_write_jsonl, read_jsonl
from crucible.core.log import utc_now_iso


class DAGState:
    """Tracks mappings between Crucible experiment nodes and Spider Chat canvas nodes."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self._mapping_path = self.state_dir / "mapping.jsonl"
        self._config_path = self.state_dir / "config.jsonl"
        self._mappings: dict[str, dict[str, Any]] = {}
        self._config: dict[str, Any] = {}

    def init(self, flow_id: str, project_name: str = "", spiderchat_url: str = "") -> None:
        """Initialize DAG state for a project."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._config = {
            "flow_id": flow_id,
            "project_name": project_name,
            "spiderchat_url": spiderchat_url,
            "created_at": utc_now_iso(),
        }
        append_jsonl(self._config_path, {"kind": "init", "ts": utc_now_iso(), "data": self._config})

    def load(self) -> None:
        """Load mappings from disk."""
        self._mappings.clear()
        for record in read_jsonl(self._mapping_path):
            cid = record.get("crucible_node_id")
            if cid:
                self._mappings[cid] = record

        # Load config
        for record in read_jsonl(self._config_path):
            if record.get("kind") == "init":
                self._config = record.get("data", {})

    @property
    def flow_id(self) -> str:
        return self._config.get("flow_id", "")

    @property
    def spiderchat_url(self) -> str:
        return self._config.get("spiderchat_url", "")

    def add_mapping(
        self,
        crucible_node_id: str,
        canvas_node_id: str,
        node_type: str = "experiment",
        status: str = "pending",
        canvas_flow_id: str = "",
        result_metric: float | None = None,
    ) -> dict[str, Any]:
        """Record a new Crucible ↔ Spider Chat node mapping."""
        now = utc_now_iso()
        record: dict[str, Any] = {
            "crucible_node_id": crucible_node_id,
            "canvas_flow_id": canvas_flow_id or self.flow_id,
            "canvas_node_id": canvas_node_id,
            "node_type": node_type,
            "status": status,
            "created_at": now,
            "synced_at": now,
        }
        if result_metric is not None:
            record["result_metric"] = result_metric
        self._mappings[crucible_node_id] = record
        append_jsonl(self._mapping_path, record)
        return record

    def update_mapping(self, crucible_node_id: str, status: str, result_metric: float | None = None) -> None:
        """Update the status (and optionally metric) of a mapping."""
        if crucible_node_id in self._mappings:
            self._mappings[crucible_node_id]["status"] = status
            self._mappings[crucible_node_id]["synced_at"] = utc_now_iso()
            if result_metric is not None:
                self._mappings[crucible_node_id]["result_metric"] = result_metric
            self._save_mappings()

    def update_status(self, crucible_node_id: str, status: str) -> None:
        """Update the status of a mapping (convenience wrapper)."""
        self.update_mapping(crucible_node_id, status)

    def get_canvas_node_id(self, crucible_node_id: str) -> str | None:
        """Look up the Spider Chat canvas node ID for a Crucible node."""
        mapping = self._mappings.get(crucible_node_id)
        return mapping["canvas_node_id"] if mapping else None

    def get_crucible_node_id(self, canvas_node_id: str) -> str | None:
        """Look up the Crucible node ID for a Spider Chat canvas node."""
        for mapping in self._mappings.values():
            if mapping.get("canvas_node_id") == canvas_node_id:
                return mapping["crucible_node_id"]
        return None

    def get_mapping(self, crucible_node_id: str) -> dict[str, Any] | None:
        """Get mapping for a specific Crucible node ID."""
        return self._mappings.get(crucible_node_id)

    def get_all_mappings(self) -> list[dict[str, Any]]:
        """Return all current mappings."""
        return list(self._mappings.values())

    def get_synced_crucible_ids(self) -> set[str]:
        """Return set of all Crucible node IDs that have been synced."""
        return set(self._mappings.keys())

    def get_synced_canvas_ids(self) -> set[str]:
        """Return set of all Spider Chat canvas node IDs that have been synced."""
        return {m["canvas_node_id"] for m in self._mappings.values()}

    def summary(self) -> dict[str, Any]:
        """Return a summary of the DAG state."""
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for m in self._mappings.values():
            s = m.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            t = m.get("node_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "flow_id": self.flow_id,
            "project_name": self._config.get("project_name", ""),
            "total_mappings": len(self._mappings),
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
        }

    def _save_mappings(self) -> None:
        """Rewrite all mappings atomically (after in-place update)."""
        atomic_write_jsonl(self._mapping_path, list(self._mappings.values()))
