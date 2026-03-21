"""Shared type definitions used across Crucible modules."""
from __future__ import annotations

from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Version store types
# ---------------------------------------------------------------------------

class VersionMeta(TypedDict, total=False):
    """Metadata for a single version of any versioned resource."""
    resource_type: str      # "experiment_design" | "research_context"
    resource_name: str      # Slug-style unique name within type
    version: int            # Monotonically increasing per resource_name
    version_id: str         # "{resource_type}/{resource_name}@v{version}"
    created_at: str         # ISO 8601 UTC
    created_by: str         # Agent identity string
    parent_version_id: str | None
    git_sha: str | None
    git_committed: bool
    summary: str
    tags: list[str]
    checksum: str           # SHA-256 of YAML content


class ExperimentDesign(TypedDict, total=False):
    """A versioned experiment design — config that can be iterated before execution."""
    name: str
    description: str
    hypothesis: str
    config: dict[str, str]  # env var overrides
    base_preset: str        # smoke | proxy | medium | promotion
    backend: str
    tags: list[str]
    family: str
    status: str             # draft | ready | running | completed | archived
    linked_run_ids: list[str]
    parent_design: str | None
    rationale: str


class ResearchContextEntry(TypedDict, total=False):
    """A versioned research context entry — knowledge artifacts."""
    name: str
    entry_type: str         # paper | idea | reference | finding | constraint
    title: str
    content: str            # markdown
    source: str             # URL or "agent-generated"
    relevance: str
    tags: list[str]
    status: str             # active | superseded | archived


# ---------------------------------------------------------------------------
# Experiment types
# ---------------------------------------------------------------------------


class ExperimentConfig(TypedDict, total=False):
    """Configuration for a single experiment to run."""
    name: str
    config: dict[str, str]  # env var overrides
    tags: list[str]
    tier: str
    backend: str
    priority: int
    wave: str | None


class ExperimentResult(TypedDict, total=False):
    """Result from a completed (or failed) experiment."""
    id: str
    name: str
    timestamp: str
    backend: str
    preset: str
    config: dict[str, str]
    result: dict[str, Any] | None  # metric keys depend on training script output
    model_bytes: int | None
    status: str  # completed | partial_recoverable | failed | timeout | killed
    tags: list[str]
    error: str | None
    failure_class: str | None
    returncode: int | None
    duration_s: float | None
    code_fingerprint: str | None


class NodeRecord(TypedDict, total=False):
    """A compute node in the fleet (pod, VM, bare metal)."""
    name: str
    node_id: str
    gpu: str
    ssh_host: str
    ssh_port: int
    user: str
    ssh_key: str
    workspace_path: str
    python_bin: str
    state: str  # new | bootstrapped | running | dead | destroyed
    env_ready: bool
    dataset_ready: bool
    git_sha: str | None
    cost_per_hr: float
    provider: str  # runpod | ssh


class QueueItem(TypedDict, total=False):
    """An experiment in the fleet queue."""
    experiment_name: str
    run_id: str
    tier: str
    backend: str
    config: dict[str, str]
    tags: list[str]
    priority: int
    wave: str | None
    assigned_node: str | None
    lease_state: str  # queued | running | completed | finished | retryable
    attempt: int
    created_at: str
    started_at: str | None
    ended_at: str | None
    result_status: str | None
