"""Tests for crucible.data.manifest — manifest loading and shard resolution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.data.manifest import (
    find_dataset,
    find_tokenizer,
    list_datasets,
    load_manifest,
    resolve_shard_paths,
    tokenizer_artifact_paths,
)


SAMPLE_MANIFEST = {
    "datasets": [
        {
            "name": "fineweb10B_sp1024",
            "stats": {"files_train": 8, "files_val": 1},
            "shard_pattern": "{dataset}_{split}_{index:06d}.bin",
            "shard_base_name": "fineweb",
        },
        {
            "name": "pile_sp2048",
            "stats": {"files_train": 4, "files_val": 1},
        },
    ],
    "tokenizers": [
        {
            "name": "fineweb_1024_bpe",
            "model_path": "tokenizers/fineweb_1024_bpe.model",
        },
    ],
}


@pytest.fixture
def manifest_dir(tmp_path: Path) -> Path:
    """Create a directory with a manifest.json."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(SAMPLE_MANIFEST), encoding="utf-8")
    return tmp_path


def test_list_datasets():
    datasets = list_datasets(SAMPLE_MANIFEST)
    assert len(datasets) == 2
    assert datasets[0]["name"] == "fineweb10B_sp1024"


def test_find_dataset_found():
    ds = find_dataset(SAMPLE_MANIFEST, "fineweb10B_sp1024")
    assert ds["name"] == "fineweb10B_sp1024"
    assert ds["stats"]["files_train"] == 8


def test_find_dataset_not_found():
    with pytest.raises(ValueError, match="not found"):
        find_dataset(SAMPLE_MANIFEST, "nonexistent")


def test_find_tokenizer_found():
    tok = find_tokenizer(SAMPLE_MANIFEST, "fineweb_1024_bpe")
    assert tok["name"] == "fineweb_1024_bpe"


def test_find_tokenizer_not_found():
    with pytest.raises(ValueError, match="not found"):
        find_tokenizer(SAMPLE_MANIFEST, "nonexistent")


def test_tokenizer_artifact_paths():
    tok = SAMPLE_MANIFEST["tokenizers"][0]
    paths = tokenizer_artifact_paths(tok)
    assert "tokenizers/fineweb_1024_bpe.model" in paths


def test_tokenizer_artifact_paths_empty():
    with pytest.raises(ValueError, match="no downloadable"):
        tokenizer_artifact_paths({"name": "empty"})


def test_resolve_shard_paths_all_shards():
    ds = SAMPLE_MANIFEST["datasets"][0]
    paths = resolve_shard_paths(ds, remote_prefix="datasets")
    assert len(paths["train"]) == 8
    assert len(paths["val"]) == 1
    assert "fineweb_train_000000.bin" in paths["train"][0]


def test_resolve_shard_paths_limited():
    ds = SAMPLE_MANIFEST["datasets"][0]
    paths = resolve_shard_paths(ds, remote_prefix="datasets", train_shards=2)
    assert len(paths["train"]) == 2
    assert len(paths["val"]) == 1


def test_resolve_shard_paths_too_many():
    ds = SAMPLE_MANIFEST["datasets"][0]
    with pytest.raises(ValueError, match="8 training shards"):
        resolve_shard_paths(ds, remote_prefix="datasets", train_shards=100)


def test_load_manifest_from_file(manifest_dir: Path):
    from crucible.core.config import DataConfig

    cfg = DataConfig(
        source="huggingface",
        repo_id="test/repo",
        local_root=str(manifest_dir),
        manifest="manifest.json",
    )
    result = load_manifest(cfg, manifest_dir, skip_download=True)
    assert len(result["datasets"]) == 2


def test_load_manifest_missing_file(tmp_path: Path):
    from crucible.core.config import DataConfig

    cfg = DataConfig(
        source="huggingface",
        repo_id="test/repo",
        local_root=str(tmp_path),
        manifest="manifest.json",
    )
    with pytest.raises(FileNotFoundError, match="manifest.json required"):
        load_manifest(cfg, tmp_path, skip_download=True)
