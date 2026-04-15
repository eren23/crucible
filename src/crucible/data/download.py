"""DataManager: download, cache, and inspect HuggingFace dataset shards.

Generalises the original ``cached_challenge_fineweb.py`` script so that any
dataset described in a manifest.json can be fetched with the same interface.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from crucible.core.config import DataConfig, ProjectConfig
from crucible.core.log import log_info, log_step, log_success, log_warn
from crucible.data.manifest import (
    find_dataset,
    find_tokenizer,
    list_datasets,
    load_manifest,
    resolve_shard_paths,
    tokenizer_artifact_paths,
)


class DataManager:
    """Manage dataset downloads from a HuggingFace repository.

    Parameters
    ----------
    config:
        Full project configuration (uses ``config.data`` internally).
    data_root:
        Override for the local data directory.  Defaults to
        ``config.project_root / config.data.local_root``.
    """

    def __init__(self, config: ProjectConfig, *, data_root: Path | None = None) -> None:
        self.config = config
        self.data_cfg: DataConfig = config.data
        self.data_root: Path = (
            data_root
            if data_root is not None
            else (config.project_root / config.data.local_root).resolve()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        variant: str,
        train_shards: int | None = None,
        *,
        include_tokenizer: bool = True,
        include_docs: bool = False,
    ) -> dict[str, Any]:
        """Download dataset shards for *variant*.

        Parameters
        ----------
        variant:
            Dataset variant name as it appears in the manifest (e.g.
            ``"fineweb10B_sp1024"``).  This is the ``name`` field of the
            dataset entry.
        train_shards:
            Number of training shards to download.  ``None`` downloads all.
        include_tokenizer:
            Also download the tokenizer artifacts linked to this dataset.
        include_docs:
            Also download ``docs_selected.jsonl`` and its sidecar file, if
            present in the repo.

        Returns
        -------
        Summary dict with counts of downloaded / skipped files.
        """
        manifest = load_manifest(self.data_cfg, self.data_root)
        dataset_entry = find_dataset(manifest, variant)

        stats = dataset_entry.get("stats") or {}
        max_train = int(stats.get("files_train", 0))

        if train_shards is not None and train_shards > max_train:
            raise ValueError(
                f"{variant} has {max_train} training shards, but {train_shards} requested"
            )

        shard_paths = resolve_shard_paths(
            dataset_entry,
            remote_prefix=self.data_cfg.remote_prefix,
            train_shards=train_shards,
        )

        total_files = len(shard_paths["val"]) + len(shard_paths["train"])
        log_step(
            f"Downloading {variant}: {len(shard_paths['train'])} train, "
            f"{len(shard_paths['val'])} val shards ({total_files} files total)"
        )

        downloaded = 0
        skipped = 0

        # Validation shards first (always download all).
        for rp in shard_paths["val"]:
            if self._fetch_file(rp):
                downloaded += 1
            else:
                skipped += 1

        # Training shards.
        for rp in shard_paths["train"]:
            if self._fetch_file(rp):
                downloaded += 1
            else:
                skipped += 1

        # Tokenizer artifacts.
        tok_downloaded = 0
        if include_tokenizer:
            tokenizer_name = dataset_entry.get("tokenizer_name")
            if tokenizer_name:
                tok_entry = find_tokenizer(manifest, tokenizer_name)
                for art_path in tokenizer_artifact_paths(tok_entry):
                    full_path = (
                        f"{self.data_cfg.remote_prefix}/{art_path}"
                        if self.data_cfg.remote_prefix
                        else art_path
                    )
                    if self._fetch_file(full_path):
                        tok_downloaded += 1

        # Optional docs.
        if include_docs:
            prefix = self.data_cfg.remote_prefix
            for doc_file in ("docs_selected.jsonl", "docs_selected.source_manifest.json"):
                rp = f"{prefix}/{doc_file}" if prefix else doc_file
                self._fetch_file(rp)

        log_success(
            f"Download complete: {downloaded} new, {skipped} cached"
            + (f", {tok_downloaded} tokenizer files" if tok_downloaded else "")
        )

        return {
            "variant": variant,
            "train_shards": len(shard_paths["train"]),
            "val_shards": len(shard_paths["val"]),
            "downloaded": downloaded,
            "skipped": skipped,
            "tokenizer_files": tok_downloaded,
        }

    def status(self) -> dict[str, Any]:
        """Return a summary of locally available data.

        Returns
        -------
        Dict mapping dataset variant names to their local status, including
        counts of train/val shards present and whether the tokenizer is
        available.
        """
        result: dict[str, Any] = {"data_root": str(self.data_root), "datasets": {}}

        try:
            manifest = load_manifest(self.data_cfg, self.data_root, skip_download=True)
        except FileNotFoundError:
            result["manifest_available"] = False
            return result

        result["manifest_available"] = True

        for ds in list_datasets(manifest):
            ds_name = ds.get("name", "")
            ds_dir = self.data_root / "datasets" / ds_name

            stats = ds.get("stats") or {}
            max_train = int(stats.get("files_train", 0))
            max_val = int(stats.get("files_val", 0))

            # Count local files.
            local_train = 0
            local_val = 0
            if ds_dir.is_dir():
                for f in ds_dir.iterdir():
                    name = f.name
                    if "_train_" in name and name.endswith(".bin"):
                        local_train += 1
                    elif "_val_" in name and name.endswith(".bin"):
                        local_val += 1

            # Check tokenizer.
            tok_name = ds.get("tokenizer_name")
            tok_ready = False
            if tok_name:
                try:
                    tok_entry = find_tokenizer(manifest, tok_name)
                    artifacts = tokenizer_artifact_paths(tok_entry)
                    tok_ready = all(
                        (self.data_root / art).is_file() for art in artifacts
                    )
                except (ValueError, KeyError):
                    tok_ready = False

            result["datasets"][ds_name] = {
                "train_shards_local": local_train,
                "train_shards_total": max_train,
                "val_shards_local": local_val,
                "val_shards_total": max_val,
                "tokenizer_ready": tok_ready,
                "complete": (local_train >= max_train and local_val >= max_val),
            }

        return result

    def list_variants(self) -> list[str]:
        """List available dataset variant names from the manifest.

        Returns an empty list if the manifest is not yet downloaded.
        """
        try:
            manifest = load_manifest(self.data_cfg, self.data_root, skip_download=True)
        except FileNotFoundError:
            log_warn("Manifest not found locally; run download to fetch it first")
            return []

        return [ds.get("name", "") for ds in list_datasets(manifest)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_path_for_remote(self, relative_path: str) -> Path:
        """Map a repo-relative path to a local filesystem path.

        Strips the configured ``remote_prefix`` if present, then places the
        file under ``data_root``.
        """
        rp = Path(relative_path)
        prefix = self.data_cfg.remote_prefix
        if prefix and rp.parts[:1] == (prefix,):
            rp = rp.relative_to(prefix)
        return self.data_root / rp

    def _fetch_file(self, relative_path: str) -> bool:
        """Download a single file from HuggingFace, returning True if new.

        If the file already exists locally, it is skipped.  The download uses
        ``hf_hub_download`` which handles its own caching; we then hard-link
        (or copy) the result into the local data tree so the training code can
        find it at a stable path.
        """
        destination = self._local_path_for_remote(relative_path)
        if destination.exists():
            return False
        if destination.is_symlink():
            destination.unlink()

        hf_hub_download = _lazy_hf_hub_download()

        remote = Path(relative_path)
        log_info(f"Fetching {relative_path}")
        cached = Path(
            hf_hub_download(
                repo_id=self.data_cfg.repo_id,
                filename=remote.name,
                subfolder=remote.parent.as_posix() if remote.parent != Path(".") else None,
                repo_type="dataset",
            )
        )
        # HF cache entries may be snapshot symlinks.  Resolve to the real blob
        # so we always materialise a real file, not a broken relative symlink.
        cached_source = cached.resolve(strict=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(cached_source, destination)
        except OSError:
            shutil.copy2(cached_source, destination)

        return True


# ---------------------------------------------------------------------------
# Lazy import
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
