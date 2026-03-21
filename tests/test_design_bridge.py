"""Tests for crucible.runner.design — design-to-config bridge."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.store import VersionStore
from crucible.runner.design import design_to_experiment_config, link_result_to_design


SAMPLE_DESIGN = {
    "name": "looped_wider",
    "description": "Test wider hidden dims",
    "hypothesis": "d_model 512 should reduce val_loss",
    "config": {"MODEL_FAMILY": "looped", "D_MODEL": "512"},
    "base_preset": "proxy",
    "backend": "torch",
    "tags": ["exploration"],
    "family": "looped",
    "status": "ready",
    "linked_run_ids": [],
    "rationale": "Sensitivity shows D_MODEL matters.",
}

SAMPLE_META = {
    "resource_type": "experiment_design",
    "resource_name": "looped_wider",
    "version": 1,
    "version_id": "experiment_design/looped_wider@v1",
    "created_at": "2025-03-21T00:00:00Z",
    "created_by": "test-agent",
}


class TestDesignToExperimentConfig:
    def test_basic_conversion(self):
        config = design_to_experiment_config(SAMPLE_DESIGN, SAMPLE_META)
        assert config["name"] == "looped_wider"
        assert config["config"] == {"MODEL_FAMILY": "looped", "D_MODEL": "512"}
        assert config["tier"] == "proxy"
        assert config["backend"] == "torch"

    def test_design_tags_included(self):
        config = design_to_experiment_config(SAMPLE_DESIGN, SAMPLE_META)
        assert "design:looped_wider" in config["tags"]
        assert "v:1" in config["tags"]
        assert "exploration" in config["tags"]

    def test_defaults_for_missing_fields(self):
        minimal = {"name": "test", "config": {"A": "1"}}
        meta = dict(SAMPLE_META, version=2)
        config = design_to_experiment_config(minimal, meta)
        assert config["tier"] == "proxy"
        assert config["backend"] == "torch"


class TestLinkResultToDesign:
    def test_links_run_id(self, tmp_path):
        store = VersionStore(tmp_path / ".crucible")
        store.create(
            "experiment_design", "test", dict(SAMPLE_DESIGN),
            summary="v1", created_by="agent",
        )

        new_meta = link_result_to_design(store, "test", "run_123")
        assert new_meta is not None
        assert new_meta["version"] == 2

        _, content = store.get_current("experiment_design", "test")
        assert "run_123" in content["linked_run_ids"]

    def test_returns_none_for_missing_design(self, tmp_path):
        store = VersionStore(tmp_path / ".crucible")
        result = link_result_to_design(store, "nonexistent", "run_123")
        assert result is None

    def test_no_duplicate_run_ids(self, tmp_path):
        store = VersionStore(tmp_path / ".crucible")
        store.create(
            "experiment_design", "test", dict(SAMPLE_DESIGN, linked_run_ids=["run_123"]),
            summary="v1", created_by="agent",
        )

        link_result_to_design(store, "test", "run_123")
        _, content = store.get_current("experiment_design", "test")
        assert content["linked_run_ids"].count("run_123") == 1
