"""Data source plugin system for crucible.

Provides a plugin architecture for registering and managing data sources
with status checking, preparation, and validation capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from crucible.core.plugin_registry import PluginRegistry

if TYPE_CHECKING:
    from pathlib import Path


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
    manifest: Optional["Path"]
    shard_count: int
    last_prepared: Optional[datetime]
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

    def search(self) -> list[SearchResult]:
        """Search for available data sources from this plugin.

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
    _DATA_SOURCE_REGISTRY.register(name, cls, source)


def list_data_sources() -> list[str]:
    """List all registered data source names.

    Returns:
        List of registered data source names.
    """
    return _DATA_SOURCE_REGISTRY.list()


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
    """Pipeline for managing multiple data sources."""

    def __init__(self) -> None:
        """Initialize the data pipeline."""
        self._sources: dict[str, type[DataSourcePlugin]] = {}

    def register_source(self, name: str, cls: type[DataSourcePlugin]) -> None:
        """Register a data source plugin with the pipeline.

        Args:
            name: Unique name for the data source.
            cls: DataSourcePlugin subclass.
        """
        self._sources[name] = cls

    def list_sources(self) -> list[str]:
        """List all registered source names.

        Returns:
            List of registered source names.
        """
        return list(self._sources.keys())

    def get_source(self, name: str, **kwargs: Any) -> DataSourcePlugin:
        """Get a data source instance by name.

        Args:
            name: Registered name of the data source.
            **kwargs: Arguments to pass to the plugin constructor.

        Returns:
            Instance of the requested DataSourcePlugin.
        """
        if name not in self._sources:
            raise KeyError(f"Data source '{name}' not found")
        return self._sources[name](**kwargs)
