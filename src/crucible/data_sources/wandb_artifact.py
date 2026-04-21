"""WandB artifact data source plugin."""
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
    DataValidationResult,
    PreparationResult,
    SearchResult,
    register_data_source,
)


class WandBArtifactSource(DataSourcePlugin):
    """Data source backed by W&B artifacts."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.entity = config["entity"]
        self.project = config["project"]
        self.artifact = config["artifact"]
        self.artifact_type = config.get("artifact_type", "dataset")
        self.local_root = Path(config.get("local_root", "./data"))
        self.api_key = config.get("api_key") or os.environ.get("WANDB_API_KEY")

    def _sanitize_path(self, s: str) -> str:
        return s.replace("..", "").replace("/", "--").replace(":", "--")

    def _get_local_artifact_path(self) -> Path:
        safe_name = self.artifact.replace(":", "--").replace("/", "--")
        return self.local_root / "wandb" / self._sanitize_path(self.entity) / self._sanitize_path(self.project) / safe_name

    def status(self) -> DataStatusResult:
        artifact_path = self._get_local_artifact_path()
        if not artifact_path.exists():
            return DataStatusResult(
                status=DataStatus.MISSING,
                manifest=None,
                shard_count={},
                last_prepared=None,
                issues=["Artifact not downloaded"],
            )

        manifest_path = artifact_path / "crucible_manifest.json"
        if not manifest_path.exists():
            return DataStatusResult(
                status=DataStatus.PARTIAL,
                manifest=None,
                shard_count={},
                last_prepared=datetime.fromtimestamp(artifact_path.stat().st_mtime),
                issues=["No manifest found in artifact"],
            )

        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            shard_files = list(artifact_path.glob("**/*.bin"))
            return DataStatusResult(
                status=DataStatus.FRESH,
                manifest=manifest,
                shard_count={"total": len(shard_files)},
                last_prepared=datetime.fromtimestamp(artifact_path.stat().st_mtime),
                issues=[],
            )
        except Exception as e:
            return DataStatusResult(
                status=DataStatus.PARTIAL,
                manifest=None,
                shard_count={},
                last_prepared=None,
                issues=[f"Failed to read manifest: {e}"],
            )

    def prepare(self, force: bool = False, background: bool = False) -> PreparationResult:
        """Download artifact from W&B."""
        job_id = str(uuid.uuid4())

        if not self.api_key:
            return PreparationResult(
                success=False,
                job_id=job_id,
                message="WANDB_API_KEY not set",
                shards_downloaded=0,
            )

        if not force and self.status().status == DataStatus.FRESH:
            return PreparationResult(
                success=True, job_id=None, message="Artifact already downloaded", shards_downloaded=0
            )

        try:
            import wandb

            api = wandb.Api(api_key=self.api_key)
            artifact_path = self._get_local_artifact_path()
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

            artifact = api.artifact(
                f"{self.entity}/{self.project}/{self.artifact}", type=self.artifact_type
            )
            artifact.download(root=str(artifact_path))

            manifest = {
                "name": self.artifact,
                "entity": self.entity,
                "project": self.project,
                "files": [f.name for f in artifact_path.glob("*")],
            }
            with open(artifact_path / "crucible_manifest.json", "w") as f:
                json.dump(manifest, f)

            shard_files = list(artifact_path.glob("**/*.bin"))
            return PreparationResult(
                success=True,
                job_id=job_id if background else None,
                message=f"Downloaded artifact with {len(shard_files)} files",
                shards_downloaded=len(shard_files),
            )
        except Exception as e:
            return PreparationResult(
                success=False, job_id=job_id, message=str(e), shards_downloaded=0
            )

    def validate(self) -> DataValidationResult:
        """Validate W&B artifact integrity."""
        artifact_path = self._get_local_artifact_path()
        if not artifact_path.exists():
            return DataValidationResult(valid=False, errors=["Artifact not downloaded"], warnings=[])

        errors = []
        warnings = []

        manifest_path = artifact_path / "crucible_manifest.json"
        if not manifest_path.exists():
            errors.append("crucible_manifest.json not found in artifact")

        shard_files = list(artifact_path.glob("**/*.bin"))
        if len(shard_files) == 0:
            warnings.append("No .bin shard files found")

        return DataValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def search(
        self, query: str, entity: str | None = None, project: str | None = None
    ) -> list[SearchResult]:
        """Search W&B for artifacts."""
        try:
            import wandb

            api = wandb.Api(api_key=self.api_key)
            search_entity = entity or self.entity
            search_project = project or self.project

            artifacts = api.artifacts(
                type=self.artifact_type, name=query, project=f"{search_entity}/{search_project}"
            )
            results = []
            for a in artifacts:
                results.append(
                    SearchResult(
                        name=a.name,
                        source="wandb",
                        description=f"W&B artifact: {a.name}",
                        shard_count=0,
                        artifact=a.name,
                    )
                )
            return results
        except Exception:
            return []


register_data_source("wandb_artifact", WandBArtifactSource, source="builtin")
