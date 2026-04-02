"""HuggingFace data source plugin."""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from crucible.core.data_sources import (
    DataSourcePlugin,
    DataStatus,
    DataStatusResult,
    PreparationResult,
    SearchResult,
    ValidationResult,
    register_data_source,
)
from crucible.data.manifest import load_manifest


class HuggingFaceDataSource(DataSourcePlugin):
    """Data source backed by HuggingFace datasets."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.repo_id = config["repo_id"]
        self.manifest_path = config.get("manifest_path", "manifest.json")
        self.local_root = Path(config.get("local_root", "./data"))
        self.token = config.get("token") or os.environ.get("HF_TOKEN")

    def _get_manifest_local_path(self) -> Path:
        return self.local_root / self.repo_id.replace("/", "--") / self.manifest_path

    def _get_shard_root(self) -> Path:
        return self.local_root / self.repo_id.replace("/", "--")

    def status(self) -> DataStatusResult:
        manifest_local = self._get_manifest_local_path()
        if not manifest_local.exists():
            return DataStatusResult(
                status=DataStatus.MISSING,
                manifest=None,
                shard_count={},
                last_prepared=None,
                issues=["Manifest not found"],
            )

        try:
            manifest = load_manifest(str(manifest_local))
            shard_root = self._get_shard_root()
            shard_files = list(shard_root.glob("**/*.bin"))
            train_shards = [f for f in shard_files if "train" in f.name]
            val_shards = [f for f in shard_files if "val" in f.name]
            return DataStatusResult(
                status=DataStatus.FRESH,
                manifest=manifest,
                shard_count={"train": len(train_shards), "val": len(val_shards)},
                last_prepared=datetime.fromtimestamp(manifest_local.stat().st_mtime),
                issues=[],
            )
        except Exception as e:
            return DataStatusResult(
                status=DataStatus.MISSING,
                manifest=None,
                shard_count={},
                last_prepared=None,
                issues=[f"Failed to load manifest: {e}"],
            )

    def prepare(self, force: bool = False, background: bool = False) -> PreparationResult:
        """Download data from HuggingFace."""
        from crucible.data import DataManager

        manifest_local = self._get_manifest_local_path()
        current_status = self.status()
        if not force and current_status.status == DataStatus.FRESH:
            return PreparationResult(success=True, job_id=None, message="Data already fresh", shards_downloaded=0)

        job_id = str(uuid.uuid4())

        try:
            dm = DataManager(
                repo_id=self.repo_id,
                local_root=str(self.local_root),
                token=self.token,
            )
            result = dm.download()
            return PreparationResult(
                success=True,
                job_id=job_id if background else None,
                message=f"Downloaded {result.get('train_shards', 0)} train shards",
                shards_downloaded=result.get("train_shards", 0),
            )
        except Exception as e:
            return PreparationResult(success=False, job_id=job_id, message=str(e), shards_downloaded=0)

    def validate(self) -> ValidationResult:
        """Validate HuggingFace data integrity."""
        manifest_local = self._get_manifest_local_path()
        if not manifest_local.exists():
            return ValidationResult(valid=False, errors=["Manifest not found"], warnings=[])

        errors = []
        warnings = []

        try:
            manifest = load_manifest(str(manifest_local))
        except Exception as e:
            return ValidationResult(valid=False, errors=[f"Manifest parse error: {e}"], warnings=[])

        shard_root = self._get_shard_root()
        shard_files = list(shard_root.glob("**/*.bin"))

        for dataset in manifest.get("datasets", []):
            expected = dataset.get("stats", {}).get("files_train", 0)
            actual = len([f for f in shard_files if "train" in f.name])
            if actual < expected:
                errors.append(f"Train shards: expected {expected}, found {actual}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def search(self, query: str) -> list[SearchResult]:
        """Search HuggingFace for datasets."""
        try:
            from huggingface_hub import list_datasets
            results = []
            for ds in list_datasets(search=query, limit=10):
                results.append(SearchResult(
                    name=ds.id,
                    source="huggingface",
                    description=ds.description or "",
                    shard_count=0,
                    repo_id=ds.id,
                ))
            return results
        except Exception:
            return []


register_data_source("huggingface", HuggingFaceDataSource, source="builtin")