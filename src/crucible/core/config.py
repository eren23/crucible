"""Load and validate crucible.yaml project configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    type: str = "ssh"  # "runpod" | "ssh"
    ssh_key: str = "~/.ssh/id_ed25519"
    image: str = ""
    gpu_types: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    source: str = "huggingface"
    repo_id: str = ""
    remote_prefix: str = "datasets"
    local_root: str = "./data"
    manifest: str = "manifest.json"


@dataclass
class TrainingConfig:
    script: str = "train.py"
    backend: str = "torch"  # "torch" | "mlx" | custom
    python: str = "python3"


@dataclass
class ResearcherConfig:
    model: str = "claude-sonnet-4-6-20250514"
    max_tokens: int = 4096
    budget_hours: float = 10.0
    max_iterations: int = 20
    program_file: str = "program.md"


@dataclass
class ProjectConfig:
    name: str = "crucible-project"
    version: str = "0.1.0"
    project_root: Path = field(default_factory=Path.cwd)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: list[TrainingConfig] = field(default_factory=list)
    presets: dict[str, dict[str, str]] = field(default_factory=dict)
    researcher: ResearcherConfig = field(default_factory=ResearcherConfig)
    sync_excludes: list[str] = field(default_factory=lambda: [
        ".git", ".venv", "__pycache__", "logs", "data/datasets", "data/tokenizers",
    ])
    results_file: str = "experiments.jsonl"
    fleet_results_file: str = "experiments_fleet.jsonl"
    logs_dir: str = "logs"
    nodes_file: str = "nodes.json"


def _build_provider(raw: dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        type=raw.get("type", "ssh"),
        ssh_key=raw.get("ssh_key", "~/.ssh/id_ed25519"),
        image=raw.get("image", ""),
        gpu_types=raw.get("gpu_types", []),
        defaults=raw.get("defaults", {}),
    )


def _build_data(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        source=raw.get("source", "huggingface"),
        repo_id=raw.get("repo_id", ""),
        remote_prefix=raw.get("remote_prefix", "datasets"),
        local_root=raw.get("local_root", "./data"),
        manifest=raw.get("manifest", "manifest.json"),
    )


def _build_training(raw: list[dict[str, Any]]) -> list[TrainingConfig]:
    return [
        TrainingConfig(
            script=t.get("script", "train.py"),
            backend=t.get("backend", "torch"),
            python=t.get("python", "python3"),
        )
        for t in raw
    ]


def _build_researcher(raw: dict[str, Any]) -> ResearcherConfig:
    return ResearcherConfig(
        model=raw.get("model", "claude-sonnet-4-6-20250514"),
        max_tokens=raw.get("max_tokens", 4096),
        budget_hours=raw.get("budget_hours", 10.0),
        max_iterations=raw.get("max_iterations", 20),
        program_file=raw.get("program_file", "program.md"),
    )


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load a crucible.yaml file and return a ProjectConfig."""
    if path is None:
        path = find_config()
    if path is None:
        return ProjectConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project_root = path.parent.resolve()

    return ProjectConfig(
        name=raw.get("name", "crucible-project"),
        version=raw.get("version", "0.1.0"),
        project_root=project_root,
        provider=_build_provider(raw.get("provider", {})),
        data=_build_data(raw.get("data", {})),
        training=_build_training(raw.get("training", [])),
        presets=raw.get("presets", {}),
        researcher=_build_researcher(raw.get("researcher", {})),
        sync_excludes=raw.get("sync_excludes", [
            ".git", ".venv", "__pycache__", "logs", "data/datasets", "data/tokenizers",
        ]),
        results_file=raw.get("results_file", "experiments.jsonl"),
        fleet_results_file=raw.get("fleet_results_file", "experiments_fleet.jsonl"),
        logs_dir=raw.get("logs_dir", "logs"),
        nodes_file=raw.get("nodes_file", "nodes.json"),
    )


def find_config() -> Path | None:
    """Walk up from cwd looking for crucible.yaml."""
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "crucible.yaml"
        if candidate.exists():
            return candidate
    return None


def generate_default_config() -> str:
    """Return a default crucible.yaml template as a string."""
    return """\
# Crucible project configuration
name: my-project
version: "0.1.0"

# Compute provider
provider:
  type: ssh                              # "runpod" | "ssh"
  ssh_key: ~/.ssh/id_ed25519
  # image: runpod/pytorch:...            # container image (RunPod only)
  # gpu_types: ["NVIDIA GeForce RTX 4090"]

# Data pipeline
data:
  source: huggingface
  repo_id: ""                            # HuggingFace dataset repo
  local_root: ./data
  manifest: manifest.json

# Training backends
training:
  - backend: torch
    script: train.py

# Experiment presets
presets:
  smoke:
    MAX_WALLCLOCK_SECONDS: "60"
    ITERATIONS: "400"
  proxy:
    MAX_WALLCLOCK_SECONDS: "1800"
    ITERATIONS: "6000"

# Autonomous researcher
researcher:
  model: claude-sonnet-4-6-20250514
  budget_hours: 10.0
  program_file: program.md

# Sync exclusions
sync_excludes:
  - .git
  - .venv
  - __pycache__
  - logs
  - data/datasets

# Output paths
results_file: experiments.jsonl
logs_dir: logs
"""
