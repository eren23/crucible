"""Tests for LocalFilesSource."""
import pytest
from pathlib import Path
from crucible.data_sources.local_files import LocalFilesSource
from crucible.core.data_sources import DataStatus


def test_local_source_initialization(tmp_path):
    source = LocalFilesSource(
        name="test_local",
        config={"path": str(tmp_path), "format": "binary"},
    )
    assert source.name == "test_local"
    assert source.path == tmp_path


def test_local_source_missing(tmp_path):
    source = LocalFilesSource(
        name="test", config={"path": str(tmp_path / "nonexistent")}
    )
    result = source.status()
    assert result.status == DataStatus.MISSING


def test_local_source_partial_no_shards(tmp_path):
    source = LocalFilesSource(name="test", config={"path": str(tmp_path)})
    result = source.status()
    assert result.status == DataStatus.PARTIAL
    assert "No shard files found" in result.issues[0]
