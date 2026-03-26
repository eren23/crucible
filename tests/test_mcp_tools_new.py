"""Tests for new MCP tools: get_run_logs, model_fetch_architecture, get_architecture_guide."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# get_run_logs
# ---------------------------------------------------------------------------


class TestGetRunLogs:
    def test_local_logs_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When local log files exist, return them without SSH."""
        from crucible.mcp.tools import get_run_logs

        # Set up project structure
        project = tmp_path / "project"
        project.mkdir()
        fleet_runs = project / "fleet_runs" / "node-1" / "logs"
        fleet_runs.mkdir(parents=True)
        (fleet_runs / "run123.launcher.txt").write_text("Launching experiment...\n")
        (fleet_runs / "run123.txt").write_text("step:1/100 train_loss:4.5\nstep:2/100 train_loss:4.3\n")

        # Mock config
        class FakeConfig:
            project_root = project
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = get_run_logs({"run_id": "run123"})
        assert result["found"] is True
        assert result["source"] == "local"
        assert "train_loss" in result["log_text"]
        assert result["lines_returned"] > 0
        assert len(result["log_files"]) == 2

    def test_local_logs_tail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Tail parameter limits returned lines."""
        from crucible.mcp.tools import get_run_logs

        project = tmp_path / "project"
        project.mkdir()
        logs_dir = project / "fleet_runs" / "node-1" / "logs"
        logs_dir.mkdir(parents=True)
        lines = "\n".join(f"step:{i}/100 train_loss:{4.5 - i * 0.01}" for i in range(50))
        (logs_dir / "run456.txt").write_text(lines)

        class FakeConfig:
            project_root = project
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = get_run_logs({"run_id": "run456", "tail_lines": 10})
        assert result["found"] is True
        assert result["lines_returned"] == 10

    def test_no_logs_no_queue(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When no local logs and run_id not in queue, return not found."""
        from crucible.mcp.tools import get_run_logs

        project = tmp_path / "project"
        project.mkdir()
        (project / "nodes.json").write_text("[]")

        class FakeConfig:
            project_root = project
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = get_run_logs({"run_id": "nonexistent"})
        assert result["found"] is False
        assert "not found in queue" in result["reason"]

    def test_project_level_logs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Logs in project-level logs/ directory are also found."""
        from crucible.mcp.tools import get_run_logs

        project = tmp_path / "project"
        project.mkdir()
        logs_dir = project / "logs"
        logs_dir.mkdir()
        (logs_dir / "run789.txt").write_text("step:1/10 train_loss:3.0\n")

        class FakeConfig:
            project_root = project
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = get_run_logs({"run_id": "run789"})
        assert result["found"] is True
        assert result["source"] == "local"


# ---------------------------------------------------------------------------
# model_fetch_architecture
# ---------------------------------------------------------------------------


class TestModelFetchArchitecture:
    def test_fetch_local_python(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Local .py architecture files are found first."""
        from crucible.mcp.tools import model_fetch_architecture

        project = tmp_path / "project"
        arch_dir = project / ".crucible" / "architectures"
        arch_dir.mkdir(parents=True)
        code = "from crucible.models.registry import register_model\nregister_model('my_arch', lambda a: None)\n"
        (arch_dir / "my_arch.py").write_text(code)

        class FakeConfig:
            project_root = project
            store_dir = ".crucible"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_fetch_architecture({"family": "my_arch"})
        assert result["family"] == "my_arch"
        assert result["kind"] == "code"
        assert result["source"] == "local"
        assert "register_model" in result["content"]

    def test_fetch_local_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Local .yaml spec files are found."""
        from crucible.mcp.tools import model_fetch_architecture

        project = tmp_path / "project"
        arch_dir = project / ".crucible" / "architectures"
        arch_dir.mkdir(parents=True)
        (arch_dir / "my_spec.yaml").write_text("name: my_spec\nbase: tied_embedding_lm\n")

        class FakeConfig:
            project_root = project
            store_dir = ".crucible"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_fetch_architecture({"family": "my_spec"})
        assert result["family"] == "my_spec"
        assert result["kind"] == "spec"
        assert result["source"] == "local"

    def test_fetch_builtin(self, monkeypatch: pytest.MonkeyPatch):
        """Builtin architectures are found as fallback."""
        from crucible.mcp.tools import model_fetch_architecture

        class FakeConfig:
            project_root = Path("/nonexistent")
            store_dir = ".crucible"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_fetch_architecture({"family": "baseline"})
        assert result["family"] == "baseline"
        assert result["source"] == "builtin"
        # Could be code or spec depending on what files exist
        assert result["kind"] in ("code", "spec")
        assert len(result["content"]) > 0

    def test_fetch_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Unknown family returns an error with available list."""
        from crucible.mcp.tools import model_fetch_architecture

        class FakeConfig:
            project_root = tmp_path
            store_dir = ".crucible"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_fetch_architecture({"family": "nonexistent_xyz"})
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_fetch_global_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Global hub YAML specs are found via hub metadata."""
        from crucible.core.hub import HubStore
        from crucible.mcp.tools import model_fetch_architecture

        hub_dir = tmp_path / "hub"
        hub = HubStore.init(hub_dir=hub_dir, name="hub")
        hub.store_architecture(
            name="global_spec",
            code="name: global_spec\nbase: tied_embedding_lm\nembedding: {}\nblock: {}\nstack: {}\n",
            kind="spec",
        )

        FakeConfig = type(
            "FakeConfig",
            (),
            {"project_root": tmp_path, "store_dir": ".crucible", "hub_dir": str(hub_dir)},
        )

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_fetch_architecture({"family": "global_spec"})
        assert result["family"] == "global_spec"
        assert result["kind"] == "spec"
        assert result["source"] == "global"
        assert "name: global_spec" in result["content"]

    def test_py_takes_precedence_over_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When both .py and .yaml exist at same scope, .py wins."""
        from crucible.mcp.tools import model_fetch_architecture

        project = tmp_path / "project"
        arch_dir = project / ".crucible" / "architectures"
        arch_dir.mkdir(parents=True)
        (arch_dir / "dual.py").write_text("# Python version\n")
        (arch_dir / "dual.yaml").write_text("# YAML version\n")

        class FakeConfig:
            project_root = project
            store_dir = ".crucible"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_fetch_architecture({"family": "dual"})
        assert result["kind"] == "code"  # .py takes precedence


class TestModelImportArchitecture:
    def test_imports_global_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from crucible.core.hub import HubStore
        from crucible.mcp.tools import model_import_architecture

        project = tmp_path / "project"
        project.mkdir()
        hub_dir = tmp_path / "hub"
        hub = HubStore.init(hub_dir=hub_dir, name="hub")
        hub.store_architecture(
            name="global_spec",
            code="name: global_spec\nbase: tied_embedding_lm\nembedding: {}\nblock: {}\nstack: {}\n",
            kind="spec",
        )

        FakeConfig = type(
            "FakeConfig",
            (),
            {"project_root": project, "store_dir": ".crucible", "hub_dir": str(hub_dir)},
        )

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = model_import_architecture({"name": "global_spec"})
        assert result["status"] == "imported"
        assert result["path"].endswith("global_spec.yaml")
        assert (project / ".crucible" / "architectures" / "global_spec.yaml").exists()


# ---------------------------------------------------------------------------
# get_architecture_guide
# ---------------------------------------------------------------------------


class TestGetArchitectureGuide:
    def test_returns_decision_tree(self):
        from crucible.mcp.tools import get_architecture_guide

        result = get_architecture_guide({})
        assert "decision_tree" in result
        assert "use_declarative_composition" in result["decision_tree"]
        assert "use_python_plugin" in result["decision_tree"]

    def test_returns_workflows(self):
        from crucible.mcp.tools import get_architecture_guide

        result = get_architecture_guide({})
        assert "workflows" in result
        assert "declarative_composition" in result["workflows"]
        assert "python_plugin" in result["workflows"]
        assert "steps" in result["workflows"]["declarative_composition"]
        assert "steps" in result["workflows"]["python_plugin"]

    def test_returns_tips(self):
        from crucible.mcp.tools import get_architecture_guide

        result = get_architecture_guide({})
        assert "tips" in result
        assert len(result["tips"]) > 0


# ---------------------------------------------------------------------------
# get_fleet_status with metrics
# ---------------------------------------------------------------------------


class TestFleetStatusMetrics:
    def test_without_metrics(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Default call without metrics returns no metrics key."""
        from crucible.mcp.tools import get_fleet_status

        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text(json.dumps([
            {"name": "n1", "node_id": "abc", "state": "running", "gpu": "A100",
             "ssh_host": "1.2.3.4", "env_ready": True, "dataset_ready": True},
        ]))

        class FakeConfig:
            project_root = tmp_path
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = get_fleet_status({})
        assert "nodes" in result
        assert "metrics" not in result

    def test_with_metrics_mocked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When include_metrics=true with mocked SSH, metrics are returned."""
        from crucible.mcp.tools import get_fleet_status, _probe_node_metrics

        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text(json.dumps([
            {"name": "n1", "node_id": "abc", "state": "running", "gpu": "A100",
             "ssh_host": "1.2.3.4", "env_ready": True, "dataset_ready": True},
        ]))

        class FakeConfig:
            project_root = tmp_path
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        # Mock _probe_node_metrics to avoid SSH
        monkeypatch.setattr(
            "crucible.mcp.tools._probe_node_metrics",
            lambda node: {"node": node["name"], "gpu_utilization_pct": 85},
        )

        result = get_fleet_status({"include_metrics": True})
        assert "metrics" in result
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["gpu_utilization_pct"] == 85


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------


class TestPreconditionChecks:
    def test_bootstrap_no_ssh_nodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """bootstrap_nodes returns actionable error when no SSH nodes exist."""
        from crucible.mcp.tools import bootstrap_nodes

        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text(json.dumps([
            {"name": "n1", "state": "provisioning"},
        ]))

        class FakeConfig:
            project_root = tmp_path
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = bootstrap_nodes({})
        assert "error" in result
        assert "fleet_refresh" in result["error"]

    def test_dispatch_no_bootstrapped_nodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """dispatch_experiments returns actionable error when no bootstrapped nodes."""
        from crucible.mcp.tools import dispatch_experiments

        nodes_file = tmp_path / "nodes.json"
        nodes_file.write_text(json.dumps([
            {"name": "n1", "state": "running", "ssh_host": "1.2.3.4"},
        ]))

        class FakeConfig:
            project_root = tmp_path
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = dispatch_experiments({})
        assert "error" in result
        assert "bootstrap_nodes" in result["error"]

    def test_collect_no_nodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """collect_results returns actionable error when no nodes exist."""
        from crucible.mcp.tools import collect_results

        class FakeConfig:
            project_root = tmp_path
            nodes_file = "nodes.json"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = collect_results({})
        assert "error" in result
        assert "provision_nodes" in result["error"]

    def test_generate_hypotheses_no_api_key(self, monkeypatch: pytest.MonkeyPatch):
        """design_generate_hypotheses returns error when ANTHROPIC_API_KEY missing."""
        from crucible.mcp.tools import design_generate_hypotheses

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = design_generate_hypotheses({})
        assert "error" in result
        assert "ANTHROPIC_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# config_get_modalities
# ---------------------------------------------------------------------------


class TestConfigGetModalities:
    def test_returns_training_backends(self, monkeypatch: pytest.MonkeyPatch):
        """config_get_modalities returns training backends with modality tags."""
        from crucible.mcp.tools import config_get_modalities
        from crucible.core.config import TrainingConfig

        class FakeConfig:
            training = [
                TrainingConfig(backend="torch", script="train.py", modality="lm"),
                TrainingConfig(backend="mlx", script="train_mlx.py", modality="lm"),
            ]

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = config_get_modalities({})
        assert "training_backends" in result
        assert len(result["training_backends"]) == 2
        assert result["training_backends"][0]["backend"] == "torch"
        assert result["training_backends"][0]["modality"] == "lm"
        assert result["training_backends"][1]["backend"] == "mlx"

    def test_returns_data_adapters(self, monkeypatch: pytest.MonkeyPatch):
        """config_get_modalities lists registered data adapters."""
        from crucible.mcp.tools import config_get_modalities
        from crucible.core.config import TrainingConfig

        class FakeConfig:
            training = [TrainingConfig(backend="torch", script="train.py")]

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = config_get_modalities({})
        assert "data_adapters" in result
        assert "token" in result["data_adapters"]

    def test_returns_objectives(self, monkeypatch: pytest.MonkeyPatch):
        """config_get_modalities lists registered training objectives."""
        from crucible.mcp.tools import config_get_modalities
        from crucible.core.config import TrainingConfig

        class FakeConfig:
            training = [TrainingConfig(backend="torch", script="train.py")]

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: FakeConfig())

        result = config_get_modalities({})
        assert "objectives" in result
        assert "cross_entropy" in result["objectives"]
        assert "mse" in result["objectives"]
