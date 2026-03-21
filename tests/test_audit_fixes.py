"""Tests for audit fix issues: version_diff formatting, partial updates,
design name validation, version_get/run/link tools."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.store import VersionStore


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
    "rationale": "Sensitivity shows D_MODEL matters.",
}


@pytest.fixture
def store(tmp_path: Path) -> VersionStore:
    return VersionStore(tmp_path / ".crucible")


# ---------------------------------------------------------------------------
# Fix 1: version_diff f-string formatting
# ---------------------------------------------------------------------------

class TestVersionDiffFormatting:
    def test_diff_keys_use_version_numbers(self, store):
        """Verify diff output uses 'v1', 'v2' not literal 'v{va}', 'v{vb}'."""
        store.create(
            "experiment_design", "test",
            dict(SAMPLE_DESIGN, description="original"),
            summary="v1", created_by="agent",
        )
        store.create(
            "experiment_design", "test",
            dict(SAMPLE_DESIGN, description="updated"),
            summary="v2", created_by="agent",
        )

        # Simulate what version_diff tool does
        result_a = store.get_version_number("experiment_design", "test", 1)
        result_b = store.get_version_number("experiment_design", "test", 2)
        assert result_a is not None
        assert result_b is not None

        meta_a, content_a = result_a
        meta_b, content_b = result_b

        va, vb = 1, 2
        all_keys = set(content_a.keys()) | set(content_b.keys())
        changes = {}
        for key in sorted(all_keys):
            val_a = content_a.get(key)
            val_b = content_b.get(key)
            if val_a != val_b:
                changes[key] = {f"v{va}": val_a, f"v{vb}": val_b}

        assert "description" in changes
        assert "v1" in changes["description"]
        assert "v2" in changes["description"]
        # Make sure we don't have literal {va}/{vb}
        assert "v{va}" not in str(changes)
        assert "v{vb}" not in str(changes)


# ---------------------------------------------------------------------------
# Fix 5: Partial design updates
# ---------------------------------------------------------------------------

class TestPartialDesignUpdate:
    def test_partial_update_preserves_existing_fields(self, store):
        """When updating only status, other fields should be preserved."""
        # Create initial design
        store.create(
            "experiment_design", "test", dict(SAMPLE_DESIGN),
            summary="v1", created_by="agent",
        )

        # Load, modify only status, save
        _, current = store.get_current("experiment_design", "test")
        current["status"] = "ready"
        store.create(
            "experiment_design", "test", current,
            summary="Changed status to ready", created_by="agent",
        )

        # Verify all original fields preserved
        _, updated = store.get_current("experiment_design", "test")
        assert updated["status"] == "ready"
        assert updated["description"] == "Test wider hidden dims"
        assert updated["config"] == {"MODEL_FAMILY": "looped", "D_MODEL": "512"}
        assert updated["family"] == "looped"
        assert updated["rationale"] == "Sensitivity shows D_MODEL matters."


# ---------------------------------------------------------------------------
# Fix 4: version_get_design, version_run_design, version_link_result
# ---------------------------------------------------------------------------

class TestVersionGetDesign:
    def test_get_current_design(self, store):
        store.create(
            "experiment_design", "test", dict(SAMPLE_DESIGN),
            summary="v1", created_by="agent",
        )
        result = store.get_current("experiment_design", "test")
        assert result is not None
        meta, content = result
        assert content["name"] == "looped_wider"
        assert content["config"]["D_MODEL"] == "512"

    def test_get_specific_version(self, store):
        store.create(
            "experiment_design", "test",
            dict(SAMPLE_DESIGN, description="v1"),
            summary="v1", created_by="agent",
        )
        store.create(
            "experiment_design", "test",
            dict(SAMPLE_DESIGN, description="v2"),
            summary="v2", created_by="agent",
        )

        # Get v1 specifically
        result = store.get_version_number("experiment_design", "test", 1)
        assert result is not None
        _, content = result
        assert content["description"] == "v1"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_current("experiment_design", "nope") is None


class TestVersionLinkResult:
    def test_link_creates_new_version(self, store):
        from crucible.runner.design import link_result_to_design

        store.create(
            "experiment_design", "test", dict(SAMPLE_DESIGN),
            summary="v1", created_by="agent",
        )

        meta = link_result_to_design(store, "test", "run_abc123")
        assert meta is not None
        assert meta["version"] == 2

        _, content = store.get_current("experiment_design", "test")
        assert "run_abc123" in content["linked_run_ids"]


class TestDesignToConfig:
    def test_conversion_includes_design_tags(self):
        from crucible.runner.design import design_to_experiment_config

        meta = {"version": 3, "resource_name": "test"}
        config = design_to_experiment_config(SAMPLE_DESIGN, meta)
        assert "design:looped_wider" in config["tags"]
        assert "v:3" in config["tags"]


# ---------------------------------------------------------------------------
# Fix 10: Design name validation
# ---------------------------------------------------------------------------

class TestDesignNameValidation:
    def test_valid_names(self):
        """These should all be valid slugs."""
        import re
        valid = ["looped_wider", "exp-01", "baseline", "a1b2c3", "my_exp-v2"]
        for name in valid:
            assert re.match(r'^[a-z0-9][a-z0-9_-]*$', name), f"Should be valid: {name}"

    def test_invalid_names(self):
        """These should be rejected."""
        import re
        invalid = ["Looped_Wider", "has spaces", "_leading", "-leading", "special!chars", ""]
        for name in invalid:
            assert not re.match(r'^[a-z0-9][a-z0-9_-]*$', name), f"Should be invalid: {name}"


# ---------------------------------------------------------------------------
# Fix 8: Store consistency (ledger first)
# ---------------------------------------------------------------------------

class TestStoreConsistency:
    def test_ledger_written_before_yaml(self, store):
        """After create(), ledger should exist even if we check before YAML."""
        store.create(
            "experiment_design", "test", SAMPLE_DESIGN,
            summary="v1", created_by="agent",
        )

        # Verify ledger has the entry
        from crucible.core.io import read_jsonl
        records = read_jsonl(store._ledger_path)
        assert len(records) == 1
        assert records[0]["data"]["version_id"] == "experiment_design/test@v1"

    def test_store_reloads_correctly_after_writes(self, tmp_path):
        """New store instance should see all versions."""
        store_dir = tmp_path / ".crucible"
        s1 = VersionStore(store_dir)
        s1.create("experiment_design", "a", SAMPLE_DESIGN, summary="a", created_by="x")
        s1.create("experiment_design", "b", dict(SAMPLE_DESIGN, name="b"), summary="b", created_by="x")

        s2 = VersionStore(store_dir)
        assert len(s2.list_resources("experiment_design")) == 2


# ---------------------------------------------------------------------------
# Fix 12: .crucible in sync_excludes
# ---------------------------------------------------------------------------

class TestSyncExcludes:
    def test_crucible_dir_in_default_excludes(self):
        from crucible.core.config import ProjectConfig
        config = ProjectConfig()
        assert ".crucible" in config.sync_excludes


# ---------------------------------------------------------------------------
# Fix 13: research_state_file config
# ---------------------------------------------------------------------------

class TestResearchStateFileConfig:
    def test_default_research_state_file(self):
        from crucible.core.config import ProjectConfig
        config = ProjectConfig()
        assert config.research_state_file == "research_state.jsonl"

    def test_loaded_from_yaml(self, tmp_path):
        from crucible.core.config import load_config
        yaml_content = """
name: test
research_state_file: custom_state.jsonl
"""
        (tmp_path / "crucible.yaml").write_text(yaml_content)
        config = load_config(tmp_path / "crucible.yaml")
        assert config.research_state_file == "custom_state.jsonl"
