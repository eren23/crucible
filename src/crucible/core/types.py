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
    execution_provider: str | None
    remote_node: str | None
    contract_status: str | None
    wandb: dict[str, Any] | None


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
    execution_provider: str | None
    contract_status: str | None
    wandb: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Knowledge & tracking types
# ---------------------------------------------------------------------------


class Finding(TypedDict, total=False):
    """A structured research finding — knowledge that may transfer across projects."""
    id: str                        # slug-style unique name
    title: str
    body: str                      # freeform markdown
    scope: str                     # "project" | "track" | "global"
    status: str                    # "draft" | "active" | "superseded" | "archived"
    confidence: float              # 0.0–1.0
    tags: list[str]
    source_project: str
    source_experiments: list[str]
    supersedes: str | None
    superseded_by: str | None
    track: str | None
    created_at: str
    created_by: str
    promoted_from: str | None


class Track(TypedDict, total=False):
    """A research track grouping related projects and findings."""
    name: str
    description: str
    tags: list[str]
    linked_projects: list[str]
    created_at: str
    active: bool


class ProjectRegistryEntry(TypedDict, total=False):
    """A project registered in the Crucible Hub."""
    name: str
    path: str
    registered_at: str
    tracks: list[str]


class ExperimentNote(TypedDict, total=False):
    """A freeform note attached to an experiment run."""
    note_id: str                   # "note_{utc_stamp}"
    run_id: str
    stage: str                     # freeform tag: "pre-run" | "during-run" | "post-run" | "analysis"
    tags: list[str]
    confidence: float | None
    supersedes: str | None
    finding_ids: list[str]
    created_by: str
    created_at: str
    file_path: str
    body: str


# ---------------------------------------------------------------------------
# Search tree types
# ---------------------------------------------------------------------------


class SearchTreeNode(TypedDict, total=False):
    """A node in a search tree over experiments."""
    node_id: str
    tree_name: str
    parent_node_id: str | None
    children: list[str]
    depth: int
    experiment_name: str
    run_id: str | None
    config: dict[str, str]
    status: str                    # pending | queued | running | completed | failed | pruned
    result_metric: float | None
    result: dict | None
    hypothesis: str
    rationale: str
    generation_method: str
    priority_score: float
    visit_count: int
    created_at: str
    completed_at: str | None
    pruned_at: str | None
    prune_reason: str | None
    tags: list[str]


class SearchTreeMeta(TypedDict, total=False):
    """Metadata for a search tree over experiments."""
    name: str
    description: str
    root_node_ids: list[str]
    expansion_policy: str
    pruning_policy: str
    expansion_config: dict
    pruning_config: dict
    primary_metric: str
    metric_direction: str
    max_depth: int
    max_nodes: int
    max_expansions_per_node: int
    status: str
    total_nodes: int
    completed_nodes: int
    pruned_nodes: int
    best_node_id: str | None
    best_metric: float | None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Community tap types
# ---------------------------------------------------------------------------


class PluginManifest(TypedDict, total=False):
    """Metadata for a plugin package in a tap."""
    name: str              # e.g. "lion"
    type: str              # e.g. "optimizers"
    version: str           # semver "1.0.0"
    description: str
    author: str
    tags: list[str]
    requires: list[str]    # pip deps beyond crucible
    benchmarks: dict[str, Any]
    tested_with: str       # e.g. "crucible>=0.2.1"


class InstalledPackage(TypedDict, total=False):
    """Record of an installed tap plugin."""
    name: str
    type: str
    version: str
    tap: str               # tap name it came from
    installed_at: str      # ISO timestamp
    sha: str               # git commit SHA of tap at install time


class TapInfo(TypedDict, total=False):
    """Metadata for a configured tap."""
    name: str
    url: str
    added_at: str
    last_synced: str
