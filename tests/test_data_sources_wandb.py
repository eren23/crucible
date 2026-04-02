"""Tests for WandBArtifactSource."""
import pytest
from crucible.data_sources.wandb_artifact import WandBArtifactSource
from crucible.core.data_sources import DataStatus


def test_wandb_source_initialization():
    source = WandBArtifactSource(
        name="test_wandb",
        config={
            "entity": "my_team",
            "project": "parameter-golf",
            "artifact": "fineweb_processed:latest",
            "artifact_type": "dataset",
            "local_root": "./data",
        },
    )
    assert source.name == "test_wandb"
    assert source.config["entity"] == "my_team"


def test_wandb_source_status_missing(tmp_path):
    source = WandBArtifactSource(
        name="test",
        config={"entity": "team", "project": "proj", "artifact": "data:latest", "local_root": str(tmp_path)},
    )
    result = source.status()
    assert result.status == DataStatus.MISSING
