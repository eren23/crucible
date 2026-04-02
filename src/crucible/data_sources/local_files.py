"""Local files data source plugin."""
from __future__ import annotations

import json
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


class LocalFilesSource(DataSourcePlugin):
    """Data source backed by local files."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.path = Path(config["path"])
        self.manifest_path = config.get("manifest_path")
        self.format = config.get("format", "binary")

    def status(self) -> DataStatusResult:
        if not self.path.exists():
            return DataStatusResult(
                status=DataStatus.MISSING,
                manifest=None,
                shard_count={},
                last_prepared=None,
                issues=["Path does not exist"],
            )

        shard_files = list(self.path.glob("**/*.bin"))
        if len(shard_files) == 0:
            shard_files = list(self.path.glob("**/*.jsonl"))

        if len(shard_files) == 0:
            return DataStatusResult(
                status=DataStatus.PARTIAL,
                manifest=None,
                shard_count={},
                last_prepared=datetime.fromtimestamp(self.path.stat().st_mtime),
                issues=["No shard files found in directory"],
            )

        manifest = None
        if self.manifest_path:
            manifest_file = self.path / self.manifest_path
            if manifest_file.exists():
                with open(manifest_file, encoding="utf-8") as f:
                    manifest = json.load(f)

        return DataStatusResult(
            status=DataStatus.FRESH,
            manifest=manifest,
            shard_count={"total": len(shard_files)},
            last_prepared=datetime.fromtimestamp(self.path.stat().st_mtime),
            issues=[],
        )

    def prepare(
        self, force: bool = False, background: bool = False
    ) -> PreparationResult:
        """Local files are already available - just validate."""
        job_id = str(uuid.uuid4())
        result = self.validate()
        return PreparationResult(
            success=result.valid,
            job_id=job_id if background else None,
            message="Local files ready" if result.valid else f"Validation failed: {result.errors}",
            shards_downloaded=self.status().shard_count.get("total", 0),
        )

    def validate(self) -> ValidationResult:
        """Validate local files."""
        errors = []
        warnings = []

        if not self.path.exists():
            return ValidationResult(
                valid=False, errors=["Path does not exist"], warnings=warnings
            )

        shard_files = list(self.path.glob("**/*.bin"))
        if len(shard_files) == 0:
            shard_files = list(self.path.glob("**/*.jsonl"))
        if len(shard_files) == 0:
            errors.append("No shard files (.bin or .jsonl) found")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def search(self, query: str) -> list[SearchResult]:
        """Search local files by walking directory."""
        results = []
        query_lower = query.lower()
        all_shards = len(list(self.path.glob("**/*.bin")))
        if all_shards == 0:
            all_shards = len(list(self.path.glob("**/*.jsonl")))
        for f in self.path.glob("**/*"):
            if f.is_file() and query_lower in f.name.lower():
                results.append(
                    SearchResult(
                        name=str(f.relative_to(self.path)),
                        source="local",
                        description=str(f),
                        shard_count=all_shards,
                    )
                )
                if len(results) >= 10:
                    break
        return results


register_data_source("local_files", LocalFilesSource, source="builtin")
