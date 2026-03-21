"""Tests for crucible.core.store — VersionStore CRUD, versioning, checksums."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.store import VersionStore


@pytest.fixture
def store(tmp_path: Path) -> VersionStore:
    return VersionStore(tmp_path / ".crucible")


SAMPLE_DESIGN = {
    "name": "looped_wider",
    "description": "Test wider hidden dims",
    "hypothesis": "d_model 512 should reduce val_loss",
    "config": {"MODEL_FAMILY": "looped", "D_MODEL": "512"},
    "base_preset": "proxy",
    "backend": "torch",
    "tags": ["exploration"],
    "family": "looped",
    "status": "draft",
    "linked_run_ids": [],
    "rationale": "Sensitivity analysis shows D_MODEL matters.",
}


class TestCreateAndGet:
    def test_create_returns_meta(self, store):
        meta = store.create(
            "experiment_design", "looped_wider", SAMPLE_DESIGN,
            summary="Initial design", created_by="test-agent",
        )
        assert meta["resource_type"] == "experiment_design"
        assert meta["resource_name"] == "looped_wider"
        assert meta["version"] == 1
        assert meta["version_id"] == "experiment_design/looped_wider@v1"
        assert meta["created_by"] == "test-agent"
        assert meta["summary"] == "Initial design"
        assert meta["checksum"]  # non-empty

    def test_get_current(self, store):
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        result = store.get_current("experiment_design", "test")
        assert result is not None
        meta, content = result
        assert meta["version"] == 1
        assert content["name"] == "looped_wider"

    def test_get_current_returns_none_for_missing(self, store):
        assert store.get_current("experiment_design", "nonexistent") is None

    def test_get_version_by_id(self, store):
        meta = store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        result = store.get_version(meta["version_id"])
        assert result is not None
        assert result[0]["version_id"] == meta["version_id"]

    def test_get_version_by_number(self, store):
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        result = store.get_version_number("experiment_design", "test", 1)
        assert result is not None
        meta, content = result
        assert meta["version"] == 1


class TestVersioning:
    def test_version_increments(self, store):
        m1 = store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        assert m1["version"] == 1

        updated = dict(SAMPLE_DESIGN, description="Updated")
        m2 = store.create(
            "experiment_design", "test", updated,
            summary="v2", created_by="agent",
        )
        assert m2["version"] == 2
        assert m2["parent_version_id"] == m1["version_id"]

    def test_current_yaml_updated(self, store):
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        updated = dict(SAMPLE_DESIGN, description="Updated version")
        store.create(
            "experiment_design", "test", updated,
            summary="v2", created_by="agent",
        )
        _, content = store.get_current("experiment_design", "test")
        assert content["description"] == "Updated version"

    def test_old_versions_preserved(self, store):
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        updated = dict(SAMPLE_DESIGN, description="v2 content")
        store.create(
            "experiment_design", "test", updated,
            summary="v2", created_by="agent",
        )

        # v1 still accessible
        result = store.get_version_number("experiment_design", "test", 1)
        assert result is not None
        _, content = result
        assert content["description"] == "Test wider hidden dims"

    def test_checksums_differ(self, store):
        m1 = store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        updated = dict(SAMPLE_DESIGN, description="Different")
        m2 = store.create(
            "experiment_design", "test", updated,
            summary="v2", created_by="agent",
        )
        assert m1["checksum"] != m2["checksum"]


class TestListResources:
    def test_list_empty(self, store):
        assert store.list_resources("experiment_design") == []

    def test_list_returns_latest(self, store):
        store.create(
            "experiment_design", "a", SAMPLE_DESIGN,
            summary="a-v1", created_by="agent",
        )
        store.create(
            "experiment_design", "b", dict(SAMPLE_DESIGN, name="b"),
            summary="b-v1", created_by="agent",
        )
        resources = store.list_resources("experiment_design")
        assert len(resources) == 2
        names = {r["resource_name"] for r in resources}
        assert names == {"a", "b"}

    def test_list_tag_filter(self, store):
        store.create(
            "experiment_design", "tagged", SAMPLE_DESIGN,
            summary="tagged", created_by="agent", tags=["special"],
        )
        store.create(
            "experiment_design", "untagged", SAMPLE_DESIGN,
            summary="untagged", created_by="agent", tags=[],
        )
        filtered = store.list_resources("experiment_design", tag="special")
        assert len(filtered) == 1
        assert filtered[0]["resource_name"] == "tagged"

    def test_list_status_filter(self, store):
        store.create(
            "experiment_design", "draft_one", dict(SAMPLE_DESIGN, status="draft"),
            summary="draft", created_by="agent",
        )
        store.create(
            "experiment_design", "ready_one", dict(SAMPLE_DESIGN, status="ready"),
            summary="ready", created_by="agent",
        )
        filtered = store.list_resources("experiment_design", status="ready")
        assert len(filtered) == 1
        assert filtered[0]["resource_name"] == "ready_one"


class TestHistory:
    def test_history_empty(self, store):
        assert store.history("experiment_design", "nonexistent") == []

    def test_history_returns_all_versions(self, store):
        for i in range(3):
            store.create(
                "experiment_design", "test", dict(SAMPLE_DESIGN, description=f"v{i+1}"),
                summary=f"version {i+1}", created_by="agent",
            )
        versions = store.history("experiment_design", "test")
        assert len(versions) == 3
        assert [v["version"] for v in versions] == [1, 2, 3]


class TestResearchContext:
    def test_create_context_entry(self, store):
        context = {
            "name": "width_matters",
            "entry_type": "finding",
            "title": "Width is more important than depth",
            "content": "Sensitivity analysis shows D_MODEL has 2x the spread of N_LAYERS.",
            "source": "agent-generated",
            "relevance": "Guides next experiment direction",
            "tags": ["sensitivity"],
            "status": "active",
        }
        meta = store.create(
            "research_context", "width_matters", context,
            summary="Finding about width vs depth", created_by="agent",
        )
        assert meta["resource_type"] == "research_context"
        assert meta["version"] == 1

    def test_list_context(self, store):
        store.create(
            "research_context", "finding1", {"name": "f1", "status": "active"},
            summary="f1", created_by="agent",
        )
        resources = store.list_resources("research_context")
        assert len(resources) == 1


class TestDiskLayout:
    def test_yaml_files_created(self, store):
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        designs_dir = store.store_dir / "designs" / "test"
        assert (designs_dir / "v1.yaml").exists()
        assert (designs_dir / "current.yaml").exists()

    def test_ledger_file_created(self, store):
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )
        assert store._ledger_path.exists()

    def test_store_reloads_from_disk(self, tmp_path):
        store_dir = tmp_path / ".crucible"
        store1 = VersionStore(store_dir)
        store1.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )

        # New store instance loads from same directory
        store2 = VersionStore(store_dir)
        result = store2.get_current("experiment_design", "test")
        assert result is not None
        assert result[0]["version"] == 1


class TestUnknownResourceType:
    def test_create_unknown_type(self, store):
        with pytest.raises(Exception):
            store.create(
                "unknown_type", "test", {},
                summary="bad", created_by="agent",
            )
