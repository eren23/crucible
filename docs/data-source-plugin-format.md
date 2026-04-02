# Community Data Source Plugin Format

This document describes how to create and share custom data source plugins for Crucible via the tap system.

## Overview

Data source plugins let you define custom ways to provide data for training - HuggingFace datasets, W&B artifacts, local files, or entirely custom sources.

## Directory Structure

Community data sources are distributed as directories within tap repositories:

```
data_sources/my_dataset/
├── plugin.yaml       # Required: plugin metadata
└── my_dataset.py   # Required: DataSourcePlugin implementation
```

## plugin.yaml

```yaml
name: my_dataset
type: data_source
version: "1.0.0"
description: Processed FineWeb dataset with custom tokenization
author: my_name
tags: [fineweb, tokenized, lm]
```

## Plugin Code

```python
from crucible.core.data_sources import (
    DataSourcePlugin, DataStatus, DataStatusResult,
    PreparationResult, ValidationResult, SearchResult,
    register_data_source,
)


class MyDatasetSource(DataSourcePlugin):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        # Your initialization here

    def status(self) -> DataStatusResult:
        """Check the current status of the data source.

        Returns:
            DataStatusResult with:
            - status: FRESH if data is available and up-to-date,
                      STALE if data exists but may be outdated,
                      MISSING if data not found,
                      PARTIAL if some data present but incomplete
            - manifest: optional dict with data manifest
            - shard_count: dict mapping split names to shard counts
            - last_prepared: datetime of last preparation
            - issues: list of issues encountered
        """
        ...

    def prepare(self, force=False, background=False) -> PreparationResult:
        """Download/cache/prepare data.

        Args:
            force: Force re-preparation even if data is fresh.
            background: Run preparation in background if supported.

        Returns:
            PreparationResult with:
            - success: bool
            - job_id: optional background job ID
            - message: status message
            - shards_downloaded: number of shards prepared
        """
        ...

    def validate(self) -> ValidationResult:
        """Validate data integrity and configuration.

        Returns:
            ValidationResult with:
            - valid: bool
            - errors: list of error messages
            - warnings: list of warning messages
        """
        ...

    def search(self) -> list[SearchResult]:
        """Search for available data sources from this plugin.

        Returns:
            List of SearchResult objects describing available data sources.
            Each SearchResult has: name, source, description, shard_count,
            repo_id, artifact.
        """
        return []


# Required at module level:
register_data_source("my_dataset", MyDatasetSource, source="global")
```

## DataStatus Values

- `FRESH` - Data is available and up-to-date
- `STALE` - Data exists but may be outdated
- `MISSING` - Data not found
- `PARTIAL` - Some data present but incomplete

## Return Types

### DataStatusResult

```python
@dataclass
class DataStatusResult:
    status: DataStatus
    manifest: Optional[dict[str, Any]] = None
    shard_count: dict[str, int] = field(default_factory=dict)
    last_prepared: Optional[datetime] = None
    issues: list[str] = field(default_factory=list)
```

### PreparationResult

```python
@dataclass
class PreparationResult:
    success: bool
    job_id: Optional[str]
    message: str
    shards_downloaded: int
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

### SearchResult

```python
@dataclass
class SearchResult:
    name: str
    source: str
    description: str
    shard_count: int
    repo_id: Optional[str] = None
    artifact: Optional[str] = None
```

## Installation via Tap

```bash
crucible tap add <tap-url>
crucible tap install my_dataset --type data_source
```

After installation, the data source is available via the data pipeline MCP tools.
