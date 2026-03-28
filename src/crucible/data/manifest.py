"""Manifest loading and parsing for HuggingFace dataset repositories.

A manifest.json describes the available datasets, their shard counts, and the
tokenizers associated with each dataset.  This module generalises the original
FineWeb-specific logic so any HuggingFace repo that ships a manifest.json can
be used with the same download/sync machinery.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.core.config import DataConfig
from crucible.core.log import log_info, log_step


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def _manifest_local_path(data_cfg: DataConfig, data_root: Path) -> Path:
    """Return the expected local path for manifest.json."""
    return data_root / data_cfg.manifest


def fetch_manifest(data_cfg: DataConfig, data_root: Path) -> Path:
    """Ensure the manifest file exists locally, downloading from HF if needed.

    Returns the local path to the manifest.
    """
    local = _manifest_local_path(data_cfg, data_root)
    if local.is_file():
        return local

    log_step(f"Downloading manifest from {data_cfg.repo_id}")
    hf_hub_download = _lazy_hf_hub_download()

    remote_path = f"{data_cfg.remote_prefix}/{data_cfg.manifest}" if data_cfg.remote_prefix else data_cfg.manifest
    rp = Path(remote_path)

    cached = Path(
        hf_hub_download(
            repo_id=data_cfg.repo_id,
            filename=rp.name,
            subfolder=rp.parent.as_posix() if rp.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    _link_or_copy(cached, local)
    log_info(f"Manifest saved to {local}")
    return local


def load_manifest(
    data_cfg: DataConfig,
    data_root: Path,
    *,
    skip_download: bool = False,
) -> dict[str, Any]:
    """Load the manifest dict, optionally fetching it first.

    Parameters
    ----------
    data_cfg:
        The ``DataConfig`` section from the project configuration.
    data_root:
        Absolute path to the local data directory.
    skip_download:
        If *True*, raise ``FileNotFoundError`` instead of downloading a
        missing manifest.
    """
    local = _manifest_local_path(data_cfg, data_root)
    if not local.is_file():
        if skip_download:
            raise FileNotFoundError(
                f"manifest.json required but not present at {local}; "
                "set skip_download=False to fetch it automatically"
            )
        fetch_manifest(data_cfg, data_root)

    return json.loads(local.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dataset / tokenizer lookups
# ---------------------------------------------------------------------------

def list_datasets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of dataset entries from a manifest."""
    return manifest.get("datasets", [])


def find_dataset(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Find a dataset entry by name.  Raises ``ValueError`` if not found."""
    for ds in list_datasets(manifest):
        if ds.get("name") == name:
            return ds
    available = [d.get("name") for d in list_datasets(manifest)]
    raise ValueError(
        f"Dataset {name!r} not found in manifest.  Available: {available}"
    )


def find_tokenizer(manifest: dict[str, Any], tokenizer_name: str) -> dict[str, Any]:
    """Find a tokenizer entry by name.  Raises ``ValueError`` if not found."""
    for tok in manifest.get("tokenizers", []):
        if tok.get("name") == tokenizer_name:
            return tok
    available = [t.get("name") for t in manifest.get("tokenizers", [])]
    raise ValueError(
        f"Tokenizer {tokenizer_name!r} not found in manifest.  Available: {available}"
    )


def tokenizer_artifact_paths(tokenizer_entry: dict[str, Any]) -> list[str]:
    """Extract downloadable artifact paths from a tokenizer entry."""
    paths: list[str] = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            paths.append(str(value))
    if not paths:
        raise ValueError(
            f"Tokenizer entry has no downloadable artifacts: {tokenizer_entry}"
        )
    return paths


# ---------------------------------------------------------------------------
# Shard path resolution
# ---------------------------------------------------------------------------

def resolve_shard_paths(
    dataset_entry: dict[str, Any],
    *,
    remote_prefix: str,
    train_shards: int | None = None,
) -> dict[str, list[str]]:
    """Build lists of remote paths for train and val shards.

    Parameters
    ----------
    dataset_entry:
        A single dataset dict from the manifest (must have ``name`` and
        ``stats`` with ``files_train`` / ``files_val``).
    remote_prefix:
        The repo-level prefix (e.g. ``"datasets"``).
    train_shards:
        How many training shards to include.  ``None`` means all available.

    Returns
    -------
    dict with keys ``"train"`` and ``"val"``, each a list of remote paths
    relative to the repo root.
    """
    if "name" not in dataset_entry:
        raise ValueError("Dataset manifest entry missing required 'name' field")
    ds_name = dataset_entry["name"]
    stats = dataset_entry.get("stats") or {}
    max_train = int(stats.get("files_train", 0))
    num_val = int(stats.get("files_val", 0))

    # Determine the shard filename pattern from the manifest entry, falling
    # back to the convention used by the original FineWeb pipeline.
    pattern = dataset_entry.get("shard_pattern", "{dataset}_{split}_{index:06d}.bin")

    if train_shards is None:
        train_shards = max_train
    if train_shards > max_train:
        raise ValueError(
            f"{ds_name} has {max_train} training shards, but {train_shards} requested"
        )

    prefix = f"{remote_prefix}/datasets/{ds_name}" if remote_prefix else f"datasets/{ds_name}"

    # Try to extract the base name (strip the fineweb10B_<variant> wrapper) for
    # the shard filename.  Default to the dataset name.
    base_name = dataset_entry.get("shard_base_name", ds_name.rsplit("_", 1)[0] if "_" in ds_name else ds_name)

    def _shard_path(split: str, idx: int) -> str:
        filename = pattern.format(
            dataset=base_name,
            split=split,
            index=idx,
        )
        return f"{prefix}/{filename}"

    val_paths = [_shard_path("val", i) for i in range(num_val)]
    train_paths = [_shard_path("train", i) for i in range(train_shards)]

    return {"train": train_paths, "val": val_paths}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lazy_hf_hub_download():
    """Import huggingface_hub lazily so it remains an optional dependency."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for dataset downloads.  "
            "Install it with: pip install huggingface-hub"
        ) from exc
    return hf_hub_download


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* to *dst*, falling back to copy if cross-device."""
    import os
    import shutil

    resolved = src.resolve(strict=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(resolved, dst)
    except OSError:
        shutil.copy2(resolved, dst)
