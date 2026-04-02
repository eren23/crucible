"""Tests for HuggingFaceDataSource."""
import pytest
from pathlib import Path
from crucible.data_sources.huggingface import HuggingFaceDataSource
from crucible.core.data_sources import DataStatus

def test_hf_source_initialization():
    source = HuggingFaceDataSource(
        name="test",
        config={
            "repo_id": "willdepueoai/parameter-golf",
            "manifest_path": "datasets/fineweb10B_sp1024",
            "local_root": "./data",
        }
    )
    assert source.name == "test"
    assert source.config["repo_id"] == "willdepueoai/parameter-golf"

def test_hf_source_manifest_path():
    source = HuggingFaceDataSource(
        name="test",
        config={
            "repo_id": "willdepueoai/parameter-golf",
        }
    )
    # Default manifest_path should be "manifest.json"
    assert source.manifest_path == "manifest.json"

def test_hf_source_status_missing(tmp_path):
    source = HuggingFaceDataSource(
        name="test",
        config={"repo_id": "nonexistent/repo", "local_root": str(tmp_path)}
    )
    result = source.status()
    assert result.status == DataStatus.MISSING