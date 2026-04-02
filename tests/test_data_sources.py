"""Tests for crucible.core.data_sources module."""

import pytest
from crucible.core.data_sources import (
    DataStatus,
    DataSourcePlugin,
    DataPipeline,
    register_data_source,
    list_data_sources,
    build_data_source,
)


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
                from crucible.core.data_sources import ValidationResult
                return ValidationResult(valid=True, errors=[], warnings=[])

        instance = DummySource()
        pipeline.register_source("dummy", instance)
        sources = pipeline.list_sources()
        assert any(s["name"] == "dummy" for s in sources)
