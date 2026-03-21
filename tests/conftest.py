"""Shared fixtures for the Crucible test suite."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SAMPLE_YAML = """\
name: test-project
version: "0.3.0"

provider:
  type: ssh
  ssh_key: ~/.ssh/id_ed25519
  gpu_types: ["NVIDIA A100"]

data:
  source: huggingface
  repo_id: test/dataset
  local_root: ./data
  manifest: manifest.json

training:
  - backend: torch
    script: train.py
  - backend: mlx
    script: train_mlx.py

presets:
  smoke:
    MAX_WALLCLOCK_SECONDS: "120"
    ITERATIONS: "800"
  custom_preset:
    MAX_WALLCLOCK_SECONDS: "300"
    ITERATIONS: "2000"

researcher:
  model: claude-sonnet-4-6-20250514
  budget_hours: 5.0
  max_iterations: 10
  program_file: program.md

sync_excludes:
  - .git
  - .venv

results_file: experiments.jsonl
fleet_results_file: experiments_fleet.jsonl
logs_dir: logs
nodes_file: nodes.json
"""


def _make_experiment_result(
    name: str,
    val_bpb: float,
    val_loss: float | None = None,
    model_bytes: int | None = None,
    status: str = "completed",
    config: dict[str, str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a sample ExperimentResult dict."""
    return {
        "id": f"run_{name}",
        "name": name,
        "timestamp": "2025-01-01T00:00:00Z",
        "backend": "torch",
        "preset": "proxy",
        "config": config or {"LR": "0.001"},
        "result": {
            "val_bpb": val_bpb,
            "val_loss": val_loss or val_bpb * 1.1,
            "steps_completed": 1000,
        },
        "model_bytes": model_bytes or 50000,
        "status": status,
        "tags": tags or [],
        "error": None,
        "failure_class": None,
        "returncode": 0,
        "duration_s": 120.0,
        "code_fingerprint": "abc123",
    }


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with crucible.yaml and supporting files."""
    yaml_path = tmp_path / "crucible.yaml"
    yaml_path.write_text(SAMPLE_YAML, encoding="utf-8")

    # Create results file with sample data
    results_path = tmp_path / "experiments.jsonl"
    results = [
        _make_experiment_result("baseline", 1.50, model_bytes=40000, config={"LR": "0.001"}),
        _make_experiment_result("exp_a", 1.45, model_bytes=45000, config={"LR": "0.002"}),
        _make_experiment_result("exp_b", 1.40, model_bytes=55000, config={"LR": "0.003"}),
        _make_experiment_result("exp_c", 1.55, status="failed", config={"LR": "0.01"}),
    ]
    with open(results_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    # Create fleet results file
    fleet_path = tmp_path / "experiments_fleet.jsonl"
    fleet_results = [
        _make_experiment_result("fleet_exp_1", 1.35, model_bytes=60000, config={"LR": "0.004"}),
    ]
    with open(fleet_path, "w", encoding="utf-8") as f:
        for rec in fleet_results:
            f.write(json.dumps(rec) + "\n")

    # Create logs directory
    (tmp_path / "logs").mkdir(exist_ok=True)

    # Create nodes file
    nodes_path = tmp_path / "nodes.json"
    sample_nodes = [
        {
            "name": "gpu-1",
            "node_id": "node_001",
            "gpu": "A100",
            "ssh_host": "192.168.1.10",
            "ssh_port": 22,
            "user": "root",
            "state": "ready",
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc123",
        },
        {
            "name": "gpu-2",
            "node_id": "node_002",
            "gpu": "A100",
            "ssh_host": "192.168.1.11",
            "ssh_port": 22,
            "user": "root",
            "state": "ready",
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc123",
        },
        {
            "name": "gpu-3",
            "node_id": "node_003",
            "gpu": "A100",
            "ssh_host": "192.168.1.12",
            "ssh_port": 22,
            "user": "root",
            "state": "unreachable",
            "env_ready": False,
            "dataset_ready": False,
            "git_sha": None,
        },
    ]
    nodes_path.write_text(json.dumps(sample_nodes), encoding="utf-8")

    return tmp_path


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """Return a list of sample experiment result dicts."""
    return [
        _make_experiment_result("baseline", 1.50, model_bytes=40000, config={"LR": "0.001"}),
        _make_experiment_result("exp_a", 1.45, model_bytes=45000, config={"LR": "0.002"}),
        _make_experiment_result("exp_b", 1.40, model_bytes=55000, config={"LR": "0.003"}),
    ]


@pytest.fixture
def sample_nodes() -> list[dict[str, Any]]:
    """Return a list of sample node dicts."""
    return [
        {
            "name": "gpu-1",
            "node_id": "n1",
            "state": "ready",
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc",
        },
        {
            "name": "gpu-2",
            "node_id": "n2",
            "state": "ready",
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc",
        },
        {
            "name": "gpu-3",
            "node_id": "n3",
            "state": "unreachable",
            "env_ready": False,
            "dataset_ready": False,
            "git_sha": None,
        },
    ]
