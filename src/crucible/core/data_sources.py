"""Data source plugin system for crucible.

Provides a plugin architecture for registering and managing data sources
with status checking, preparation, and validation capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from crucible.core.config import DataConfig
from crucible.core.plugin_registry import PluginRegistry

if TYPE_CHECKING:
    from pathlib import Path

# Default HuggingFace repo for the Parameter Golf dataset, used as a fallback
# when `data.repo_id` is not set in the project spec. This exists because
# Crucible was born from the Parameter Golf competition; it is NOT a general
# default and should be overridden via `data.repo_id` in crucible.yaml for any
# non-Parameter-Golf project (see docs/getting-started.md).
DEFAULT_PARAMETER_GOLF_HF_REPO = "willdepueoai/parameter-golf"


class DataStatus(Enum):
    """Status of a data source."""

    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"
    PARTIAL = "partial"


@dataclass
class DataStatusResult:
    """Result of checking data source status."""

    status: DataStatus
    manifest: Optional[dict[str, Any]]
    shard_count: dict[str, int] = field(default_factory=dict)
    last_prepared: Optional[datetime] = None
    issues: list[str] = field(default_factory=list)


@dataclass
class PreparationResult:
    """Result of preparing a data source."""

    success: bool
    job_id: Optional[str]
    message: str
    shards_downloaded: int


@dataclass
class ValidationResult:
    """Result of validating a data source configuration."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Result of searching for available data sources."""

    name: str
    source: str
    description: str
    shard_count: int
    repo_id: Optional[str] = None
    artifact: Optional[str] = None


class DataSourcePlugin(ABC):
    """Abstract base class for data source plugins.

    Implementors must provide status(), prepare(), and validate() methods.
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def status(self) -> DataStatusResult:
        """Check the current status of the data source."""
        ...

    @abstractmethod
    def prepare(self, force: bool = False, background: bool = False) -> PreparationResult:
        """Prepare the data source for use.

        Args:
            force: Force re-preparation even if data is fresh.
            background: Run preparation in background if supported.
        """
        ...

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate the data source configuration and accessibility."""
        ...

    def search(self, query: str = "") -> list[SearchResult]:
        """Search for available data sources from this plugin.

        Args:
            query: Search query to filter data sources.

        Returns:
            List of SearchResult objects describing available data sources.
        """
        return []


# Plugin registry for data sources
_DATA_SOURCE_REGISTRY = PluginRegistry[DataSourcePlugin]("data_source")


def register_data_source(name: str, cls: type[DataSourcePlugin], source: str = "builtin") -> None:
    """Register a data source plugin.

    Args:
        name: Unique name for the data source.
        cls: DataSourcePlugin subclass.
        source: Source of the plugin (builtin, global, local).
    """
    _DATA_SOURCE_REGISTRY.register(name, cls, source=source)


def list_data_sources() -> list[str]:
    """List all registered data source names.

    Returns:
        List of registered data source names.
    """
    return _DATA_SOURCE_REGISTRY.list_plugins()


def describe_data_source(name: str) -> Optional[dict[str, str]]:
    """Return metadata for a registered data source, or None if not found.

    Returns:
        {"name": str, "type": str, "source": "builtin"|"global"|"local"} or None
    """
    return _DATA_SOURCE_REGISTRY.describe_plugin(name)


def bootstrap_data_source_spec_from_data_config(
    data: DataConfig,
    *,
    plugin_name_override: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Map project ``data:`` settings to a registered plugin name and constructor config.

    Returns ``None`` when the source is not supported for fleet bootstrap pre-check
    or required fields are missing (e.g. ``local_files`` without ``path``).
    """
    plugin = (plugin_name_override or data.source or "huggingface").strip()
    cfg: dict[str, Any]

    if plugin == "huggingface":
        cfg = {
            "repo_id": data.repo_id or DEFAULT_PARAMETER_GOLF_HF_REPO,
            "variant": data.variant,
            "remote_prefix": data.remote_prefix,
            "manifest_path": data.manifest,
            "local_root": data.local_root,
        }
    elif plugin == "local_files":
        path = (data.path or "").strip()
        if not path:
            return None
        cfg = {"path": path, "format": "binary"}
        if (data.manifest or "").strip():
            cfg["manifest_path"] = data.manifest
    elif plugin == "wandb_artifact":
        if not (
            (data.wandb_entity or "").strip()
            and (data.wandb_project or "").strip()
            and (data.wandb_artifact or "").strip()
        ):
            return None
        cfg = {
            "entity": data.wandb_entity,
            "project": data.wandb_project,
            "artifact": data.wandb_artifact,
            "artifact_type": data.wandb_artifact_type,
            "local_root": data.local_root,
        }
    else:
        return None

    if config_override:
        cfg = {**cfg, **config_override}
    return plugin, cfg


def build_data_source(name: str, **kwargs: Any) -> DataSourcePlugin:
    """Build a data source plugin instance by name.

    Args:
        name: Registered name of the data source.
        **kwargs: Arguments to pass to the plugin constructor.

    Returns:
        Instance of the requested DataSourcePlugin.
    """
    return _DATA_SOURCE_REGISTRY.build(name, **kwargs)


class DataPipeline:
    """Orchestrates data sources, tracks state, and manages preparation."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSourcePlugin] = {}

    def register_source(self, name: str, plugin: DataSourcePlugin) -> None:
        self._sources[name] = plugin

    def list_sources(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "type": type(plugin).__name__}
            for name, plugin in self._sources.items()
        ]

    def get_source(self, name: str) -> Optional[DataSourcePlugin]:
        return self._sources.get(name)
