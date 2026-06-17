"""Built-in data source plugins."""
from __future__ import annotations

from crucible.data_sources.huggingface import HuggingFaceDataSource
from crucible.data_sources.local_files import LocalFilesSource
from crucible.data_sources.wandb_artifact import WandBArtifactSource

__all__ = [
    "HuggingFaceDataSource",
    "WandBArtifactSource",
    "LocalFilesSource",
]