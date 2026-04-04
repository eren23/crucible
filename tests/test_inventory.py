"""Tests for crucible.fleet.inventory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.fleet.inventory import (
    load_nodes,
    load_nodes_if_exists,
    save_nodes,
    ready_state,
    summarize_nodes,
    merge_node_record,
    merge_node_snapshots,
    upsert_node_record,
    count_bootstrapped_ready,
    next_node_index,
    classify_health,
    BAD_API_STATES,
    PAUSED_STATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ready_node(name: str, node_id: str = "n1") -> dict[str, Any]:
    return {
        "name": name,
        "node_id": node_id,
        "state": "ready",
        "env_ready": True,
        "dataset_ready": True,
        "git_sha": "abc123",
        "ssh_host": "192.168.1.1",
        "ssh_port": 22,
    }


def _unready_node(name: str, node_id: str = "n2") -> dict[str, Any]:
    return {
        "name": name,
        "node_id": node_id,
        "state": "new",
        "env_ready": False,
        "dataset_ready": False,
        "git_sha": None,
    }


# ---------------------------------------------------------------------------
# load_nodes / save_nodes
# ---------------------------------------------------------------------------

class TestLoadSaveNodes:
    def test_load_nodes(self, tmp_path):
        path = tmp_path / "nodes.json"
        nodes = [_ready_node("gpu-1")]
        path.write_text(json.dumps(nodes), encoding="utf-8")
        loaded = load_nodes(path)
        assert len(loaded) == 1
        assert loaded[0]["name"] == "gpu-1"

    def test_load_nodes_missing_file_raises(self, tmp_path):
        from crucible.core.errors import FleetError
        with pytest.raises(FleetError):
            load_nodes(tmp_path / "missing.json")

    def test_load_nodes_invalid_format_raises(self, tmp_path):
        from crucible.core.errors import FleetError
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(FleetError):
            load_nodes(path)

    def test_load_nodes_if_exists_missing(self, tmp_path):
        assert load_nodes_if_exists(tmp_path / "missing.json") == []

    def test_load_nodes_if_exists_present(self, tmp_path):
        path = tmp_path / "nodes.json"
        path.write_text(json.dumps([_ready_node("gpu-1")]), encoding="utf-8")
        loaded = load_nodes_if_exists(path)
        assert len(loaded) == 1

    def test_save_nodes(self, tmp_path):
        path = tmp_path / "nodes.json"
        nodes = [_ready_node("gpu-1"), _unready_node("gpu-2")]
        save_nodes(path, nodes)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# ready_state
# ---------------------------------------------------------------------------

class TestReadyState:
    def test_fully_ready(self):
        node = _ready_node("gpu-1")
        assert ready_state(node) == "ready"

    def test_env_missing(self):
        node = _ready_node("gpu-1")
        node["env_ready"] = False
        assert ready_state(node) == "env_missing"

    def test_dataset_missing(self):
        node = _ready_node("gpu-1")
        node["dataset_ready"] = False
        assert ready_state(node) == "dataset_missing"

    def test_unsynced(self):
        node = _ready_node("gpu-1")
        node["git_sha"] = None
        assert ready_state(node) == "unsynced"

    def test_unsynced_empty_string(self):
        node = _ready_node("gpu-1")
        node["git_sha"] = ""
        assert ready_state(node) == "unsynced"

    def test_custom_state_passthrough(self):
        node = _ready_node("gpu-1")
        node["state"] = "bootstrapping"
        assert ready_state(node) == "bootstrapping"


# ---------------------------------------------------------------------------
# summarize_nodes
# ---------------------------------------------------------------------------

class TestSummarizeNodes:
    def test_counts(self, sample_nodes):
        summary = summarize_nodes(sample_nodes)
        assert summary["nodes_total"] == 3
        assert summary["nodes_ready"] == 2
        assert summary["nodes_healthy"] == 2

    def test_empty_nodes(self):
        summary = summarize_nodes([])
        assert summary["nodes_total"] == 0
        assert summary["nodes_ready"] == 0

    def test_counts_failed(self):
        nodes = [
            _ready_node("gpu-1"),
            {**_unready_node("gpu-2"), "state": "unreachable"},
            {**_unready_node("gpu-3"), "state": "terminated"},
        ]
        summary = summarize_nodes(nodes)
        assert summary["nodes_failed"] == 2
        assert summary["nodes_unhealthy"] == 2

    def test_counts_bootstrapping(self):
        nodes = [
            {**_unready_node("gpu-1"), "state": "bootstrapping"},
            {**_unready_node("gpu-2"), "state": "creating"},
        ]
        summary = summarize_nodes(nodes)
        assert summary["nodes_bootstrapping"] == 2

    def test_counts_replaced(self):
        node = _ready_node("gpu-1")
        node["replacement"] = True
        summary = summarize_nodes([node])
        assert summary["nodes_replaced"] == 1

    def test_counts_bootstrapped(self):
        nodes = [_ready_node("gpu-1"), _unready_node("gpu-2")]
        summary = summarize_nodes(nodes)
        assert summary["nodes_bootstrapped"] == 1

    def test_summarize_counts_stopped_separately(self):
        """Stopped nodes count in nodes_stopped but NOT in nodes_failed."""
        nodes = [
            _ready_node("gpu-1"),
            {**_ready_node("gpu-2"), "state": "stopped"},
            {**_unready_node("gpu-3"), "state": "terminated"},
        ]
        summary = summarize_nodes(nodes)
        assert summary["nodes_stopped"] == 1
        assert summary["nodes_failed"] == 1  # only 'terminated', not 'stopped'
        assert summary["nodes_ready"] == 1


# ---------------------------------------------------------------------------
# merge_node_record
# ---------------------------------------------------------------------------

class TestMergeNodeRecord:
    def test_none_existing_returns_copy(self):
        incoming = _ready_node("gpu-1")
        merged = merge_node_record(None, incoming)
        assert merged == incoming
        assert merged is not incoming  # Should be a copy

    def test_merge_preserves_env_ready(self):
        existing = _ready_node("gpu-1")
        incoming = {**_ready_node("gpu-1"), "env_ready": False}
        merged = merge_node_record(existing, incoming)
        assert merged["env_ready"] is True  # Preserved from existing

    def test_merge_preserves_dataset_ready(self):
        existing = _ready_node("gpu-1")
        incoming = {**_ready_node("gpu-1"), "dataset_ready": False}
        merged = merge_node_record(existing, incoming)
        assert merged["dataset_ready"] is True

    def test_merge_sets_ready_when_fully_bootstrapped(self):
        existing = {**_ready_node("gpu-1"), "state": "bootstrapping"}
        incoming = {
            **_ready_node("gpu-1"),
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc",
        }
        merged = merge_node_record(existing, incoming)
        assert merged["state"] == "ready"

    def test_merge_respects_bad_api_states(self):
        existing = _ready_node("gpu-1")
        incoming = {**_ready_node("gpu-1"), "api_state": "terminated"}
        merged = merge_node_record(existing, incoming)
        # Even if fully bootstrapped, api_state in BAD_API_STATES prevents "ready"
        assert merged["state"] != "ready" or merged.get("api_state") == "terminated"

    def test_merge_preserves_project_metadata(self):
        existing = {**_ready_node("gpu-1"), "project": "lewm", "workspace_path": "/workspace/le-wm"}
        incoming = {k: v for k, v in _ready_node("gpu-1").items() if k not in {"project", "workspace_path"}}
        merged = merge_node_record(existing, incoming)
        assert merged["project"] == "lewm"
        assert merged["workspace_path"] == "/workspace/le-wm"

    # --- PAUSED_STATES guard -------------------------------------------------

    def test_merge_stopped_api_returns_stopped_state(self):
        """incoming api_state='stopped', existing state='ready' -> merged state='stopped'."""
        existing = _ready_node("gpu-1")
        incoming = {**_ready_node("gpu-1"), "api_state": "stopped"}
        merged = merge_node_record(existing, incoming)
        assert merged["state"] == "stopped"

    def test_merge_stopped_preserves_bootstrap_flags(self):
        """incoming api_state='stopped', existing env_ready=True -> preserved via early return."""
        existing = {**_ready_node("gpu-1"), "env_ready": True, "dataset_ready": True, "git_sha": "abc"}
        incoming = {**_unready_node("gpu-1"), "api_state": "stopped"}
        merged = merge_node_record(existing, incoming)
        assert merged["env_ready"] is True
        assert merged["dataset_ready"] is True
        assert merged["git_sha"] == "abc"

    def test_merge_stopped_does_not_promote_to_ready(self):
        """Fully bootstrapped + api_state='stopped' -> state='stopped', NOT 'ready'."""
        existing = {
            **_ready_node("gpu-1"),
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc",
        }
        incoming = {
            **_ready_node("gpu-1"),
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc",
            "api_state": "stopped",
        }
        merged = merge_node_record(existing, incoming)
        assert merged["state"] == "stopped"

    def test_merge_starting_not_clobbered_by_stopped_api(self):
        """existing state='starting', incoming api_state='stopped' -> NOT 'stopped'."""
        existing = {**_unready_node("gpu-1"), "state": "starting"}
        incoming = {**_unready_node("gpu-1"), "api_state": "stopped", "state": "starting"}
        merged = merge_node_record(existing, incoming)
        assert merged["state"] != "stopped"


# ---------------------------------------------------------------------------
# merge_node_snapshots
# ---------------------------------------------------------------------------

class TestMergeNodeSnapshots:
    def test_merge_updates_existing(self):
        existing = [_ready_node("gpu-1", node_id="n1")]
        incoming = [{**_ready_node("gpu-1", node_id="n1"), "git_sha": "new_sha"}]
        merged = merge_node_snapshots(existing, incoming)
        assert len(merged) == 1
        assert merged[0]["git_sha"] == "new_sha"

    def test_merge_keeps_orphans(self):
        existing = [_ready_node("gpu-1", node_id="n1"), _ready_node("gpu-2", node_id="n2")]
        incoming = [_ready_node("gpu-1", node_id="n1")]
        merged = merge_node_snapshots(existing, incoming)
        assert len(merged) == 2

    def test_merge_adds_new(self):
        existing = [_ready_node("gpu-1", node_id="n1")]
        incoming = [_ready_node("gpu-1", node_id="n1"), _ready_node("gpu-3", node_id="n3")]
        merged = merge_node_snapshots(existing, incoming)
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# upsert_node_record
# ---------------------------------------------------------------------------

class TestUpsertNodeRecord:
    def test_insert_new(self, tmp_path):
        path = tmp_path / "nodes.json"
        path.write_text("[]", encoding="utf-8")
        node = _ready_node("gpu-1")
        nodes = upsert_node_record(path, node)
        assert len(nodes) == 1

    def test_update_existing_by_node_id(self, tmp_path):
        path = tmp_path / "nodes.json"
        path.write_text(json.dumps([_ready_node("gpu-1", node_id="n1")]), encoding="utf-8")
        updated = {**_ready_node("gpu-1", node_id="n1"), "git_sha": "updated"}
        nodes = upsert_node_record(path, updated)
        assert len(nodes) == 1
        assert nodes[0]["git_sha"] == "updated"

    def test_update_existing_by_name(self, tmp_path):
        path = tmp_path / "nodes.json"
        existing = _ready_node("gpu-1")
        del existing["node_id"]
        path.write_text(json.dumps([existing]), encoding="utf-8")
        updated = {**existing, "git_sha": "new_sha"}
        nodes = upsert_node_record(path, updated)
        assert len(nodes) == 1
        assert nodes[0]["git_sha"] == "new_sha"


# ---------------------------------------------------------------------------
# count_bootstrapped_ready / next_node_index
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_count_bootstrapped_ready(self):
        nodes = [_ready_node("gpu-1"), _ready_node("gpu-2"), _unready_node("gpu-3")]
        assert count_bootstrapped_ready(nodes) == 2

    def test_next_node_index_empty(self):
        assert next_node_index([], "gpu") == 1

    def test_next_node_index_with_existing(self):
        nodes = [
            {"name": "gpu-1"},
            {"name": "gpu-3"},
            {"name": "gpu-5"},
        ]
        assert next_node_index(nodes, "gpu") == 6

    def test_next_node_index_different_prefix(self):
        nodes = [
            {"name": "gpu-1"},
            {"name": "cpu-2"},
        ]
        assert next_node_index(nodes, "gpu") == 2
        assert next_node_index(nodes, "cpu") == 3


# ---------------------------------------------------------------------------
# classify_health — stopped / paused pods
# ---------------------------------------------------------------------------

class TestClassifyHealthStopped:
    """Verify classify_health returns 'stopped' for paused pods, not a BAD_API_STATE."""

    def test_classify_stopped_node_returns_stopped(self):
        """Node with state='stopped' returns 'stopped', not a BAD_API_STATE label."""
        node = {**_ready_node("gpu-1"), "state": "stopped", "api_state": "stopped"}
        assert classify_health(node) == "stopped"

    def test_classify_exited_returns_stopped(self):
        """Node with api_state='exited' returns 'stopped' (paused, not failed)."""
        node = {**_ready_node("gpu-1"), "state": "exited", "api_state": "exited"}
        assert classify_health(node) == "stopped"
