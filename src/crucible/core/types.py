"""Shared type definitions used across Crucible modules."""
from __future__ import annotations

from typing import Any, TypedDict


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
    result: dict[str, Any] | None  # val_loss, val_bpb, steps_completed, etc.
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
