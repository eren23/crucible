"""HuggingFace data source plugin."""
from __future__ import annotations

import json
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


class HuggingFaceDataSource(DataSourcePlugin):
    """Data source backed by HuggingFace datasets."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.repo_id = config["repo_id"]
        self.variant = config.get("variant", "fineweb10B_sp1024")
        self.remote_prefix = config.get("remote_prefix", "datasets")
        self.manifest_path = config.get("manifest_path", "manifest.json")
        self.local_root = Path(config.get("local_root", "./data"))
        self.token = config.get("token") or os.environ.get("HF_TOKEN")

    def _get_manifest_local_path(self) -> Path:
        safe_repo = self.repo_id.replace("/", "--")
        return self.local_root / safe_repo / self.manifest_path

    def _get_shard_root(self) -> Path:
        safe_repo = self.repo_id.replace("/", "--")
        return self.local_root / safe_repo / self.remote_prefix / self.variant

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
            manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
            shard_root = self._get_shard_root()
            if shard_root.exists():
                shard_files = list(shard_root.glob("**/*.bin"))
                train_shards = [f for f in shard_files if "train" in f.name]
                val_shards = [f for f in shard_files if "val" in f.name]
                shard_count = {"train": len(train_shards), "val": len(val_shards)}
            else:
                shard_count = {}
            return DataStatusResult(
                status=DataStatus.FRESH,
                manifest=manifest,
                shard_count=shard_count,
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
        """Download data from HuggingFace using huggingface_hub directly."""
        job_id = str(uuid.uuid4())

        current_status = self.status()
        if not force and current_status.status == DataStatus.FRESH:
            return PreparationResult(
                success=True, job_id=None, message="Data already fresh", shards_downloaded=0
            )

        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            # Download manifest first
            manifest_local = self._get_manifest_local_path()
            manifest_local.parent.mkdir(parents=True, exist_ok=True)

            downloaded_manifest = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.manifest_path,
                token=self.token,
                force_download=force,
            )
            import shutil
            shutil.copy(downloaded_manifest, manifest_local)

            # Read manifest to get dataset info
            manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
            dataset_entry = None
            for ds in manifest.get("datasets", []):
                if ds.get("name") == self.variant:
                    dataset_entry = ds
                    break

            if not dataset_entry:
                return PreparationResult(
                    success=False,
                    job_id=job_id,
                    message=f"Variant {self.variant} not found in manifest",
                    shards_downloaded=0,
                )

            # Download shard files
            shard_root = self._get_shard_root()
            shard_root.mkdir(parents=True, exist_ok=True)

            shard_prefix = dataset_entry.get("path", "")
            all_files = list_repo_files(self.repo_id, token=self.token)
            shard_files = [f for f in all_files if shard_prefix in f and f.endswith(".bin")]

            for shard_file in shard_files:
                local_path = shard_root / Path(shard_file).name
                if not local_path.exists() or force:
                    downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=shard_file,
                        token=self.token,
                        force_download=force,
                    )
                    shutil.copy(downloaded, local_path)

            return PreparationResult(
                success=True,
                job_id=job_id if background else None,
                message=f"Downloaded {len(shard_files)} shard files for {self.variant}",
                shards_downloaded=len(shard_files),
            )
        except Exception as e:
            return PreparationResult(
                success=False, job_id=job_id, message=str(e), shards_downloaded=0
            )

    def validate(self) -> ValidationResult:
        """Validate HuggingFace data integrity."""
        manifest_local = self._get_manifest_local_path()
        if not manifest_local.exists():
            return ValidationResult(valid=False, errors=["Manifest not found"], warnings=[])

        errors = []
        warnings = []

        try:
            manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
        except Exception as e:
            return ValidationResult(
                valid=False, errors=[f"Manifest parse error: {e}"], warnings=[]
            )

        shard_root = self._get_shard_root()
        if shard_root.exists():
            shard_files = list(shard_root.glob("**/*.bin"))
        else:
            shard_files = []

        for dataset in manifest.get("datasets", []):
            if dataset.get("name") != self.variant:
                continue
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
                results.append(
                    SearchResult(
                        name=ds.id,
                        source="huggingface",
                        description=ds.description or "",
                        shard_count=0,
                        repo_id=ds.id,
                    )
                )
            return results
        except Exception:
            return []


register_data_source("huggingface", HuggingFaceDataSource, source="builtin")