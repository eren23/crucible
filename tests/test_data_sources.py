"""Tests for crucible.core.data_sources module."""

import pytest
import crucible.data_sources  # noqa: F401 — register builtins
from crucible.core.config import DataConfig
from crucible.core.data_sources import (
    DataStatus,
    DataSourcePlugin,
    DataPipeline,
    DEFAULT_PARAMETER_GOLF_HF_REPO,
    bootstrap_data_source_spec_from_data_config,
    describe_data_source,
    register_data_source,
    list_data_sources,
    build_data_source,
)


class TestBootstrapDataSourceSpec:
    def test_huggingface_from_data_config(self):
        data = DataConfig(
            source="huggingface",
            repo_id="org/ds",
            variant="fineweb10B_sp1024",
            manifest="manifest.json",
            local_root="./data",
        )
        name, cfg = bootstrap_data_source_spec_from_data_config(data)
        assert name == "huggingface"
        assert cfg["repo_id"] == "org/ds"
        assert cfg["variant"] == "fineweb10B_sp1024"
        assert cfg["manifest_path"] == "manifest.json"
        assert cfg["local_root"] == "./data"

    def test_huggingface_empty_repo_uses_default(self):
        data = DataConfig(source="huggingface", repo_id="")
        name, cfg = bootstrap_data_source_spec_from_data_config(data)
        assert cfg["repo_id"] == DEFAULT_PARAMETER_GOLF_HF_REPO

    def test_config_override_merges(self):
        data = DataConfig(source="huggingface", repo_id="a/b")
        name, cfg = bootstrap_data_source_spec_from_data_config(
            data, config_override={"variant": "custom"}
        )
        assert name == "huggingface"
        assert cfg["repo_id"] == "a/b"
        assert cfg["variant"] == "custom"

    def test_local_files_without_path_returns_none(self):
        data = DataConfig(source="local_files", path="")
        assert bootstrap_data_source_spec_from_data_config(data) is None

    def test_local_files_with_path(self):
        data = DataConfig(source="local_files", path="/tmp/shards")
        name, cfg = bootstrap_data_source_spec_from_data_config(data)
        assert name == "local_files"
        assert cfg["path"] == "/tmp/shards"


class TestDescribeDataSource:
    def test_describe_huggingface_builtin(self):
        info = describe_data_source("huggingface")
        assert info is not None
        assert info == {
            "name": "huggingface",
            "type": "HuggingFaceDataSource",
            "source": "builtin",
        }


class TestDataStatusEnum:
    """Test DataStatus enum values."""

    def test_data_status_fresh_value(self):
        assert DataStatus.FRESH.value == "fresh"

    def test_data_status_stale_value(self):
        assert DataStatus.STALE.value == "stale"

    def test_data_status_missing_value(self):
        assert DataStatus.MISSING.value == "missing"

    def test_data_status_partial_value(self):
        assert DataStatus.PARTIAL.value == "partial"


class TestDataSourcePluginBase:
    """Test that DataSourcePlugin cannot be instantiated directly."""

    def test_data_source_plugin_is_abstract(self):
        with pytest.raises(TypeError):
            DataSourcePlugin()


class TestDataPipeline:
    """Test DataPipeline functionality."""

    def test_pipeline_initialization(self):
        pipeline = DataPipeline()
        assert pipeline.list_sources() == []

    def test_pipeline_register_and_list_source(self):
        pipeline = DataPipeline()

        class DummySource(DataSourcePlugin):
            def status(self):
                from crucible.core.data_sources import DataStatusResult
                return DataStatusResult(
                    status=DataStatus.FRESH,
                    manifest=None,
                    shard_count={},
                    last_prepared=None,
                    issues=[],
                )

            def prepare(self, force=False, background=False):
                from crucible.core.data_sources import PreparationResult
                return PreparationResult(
                    success=True,
                    job_id=None,
                    message="ok",
                    shards_downloaded=0,
                )

            def validate(self):
                from crucible.core.data_sources import DataValidationResult
                return DataValidationResult(valid=True, errors=[], warnings=[])

        instance = DummySource(name="dummy", config={})
        pipeline.register_source("dummy", instance)
        sources = pipeline.list_sources()
        assert any(s["name"] == "dummy" for s in sources)
