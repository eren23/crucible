"""Tests for MCP external project tools."""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from crucible.mcp.tools import (
    _make_launch_id,
    bootstrap_project_tool,
    collect_project_results,
    get_project_run_status,
    list_projects,
    provision_project,
)


def _write_spec(tmp_path, name, content):
    d = tmp_path / ".crucible" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(content), encoding="utf-8")


class TestListProjects:
    def test_lists_specs(self, tmp_path, monkeypatch):
        _write_spec(tmp_path, "proj1", {"name": "proj1", "repo": "r1", "launcher": "demo_launcher", "train": "t1"})
        _write_spec(tmp_path, "proj2", {"name": "proj2", "repo": "r2", "train": "t2"})

        # Mock _get_config to return our tmp_path
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = list_projects({})

        assert len(result["projects"]) == 2
        names = {p["name"] for p in result["projects"]}
        assert names == {"proj1", "proj2"}
        proj1 = next(p for p in result["projects"] if p["name"] == "proj1")
        assert proj1["launcher"] == "demo_launcher"

    def test_empty_dir(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = list_projects({})
        assert result["projects"] == []


class TestCollectProjectResults:
    def test_missing_run_id(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        # Create the projects dir so load doesn't fail
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = collect_project_results({"run_id": "nonexistent_123"})
        assert "error" in result
        assert "No run found" in result["error"]

    def test_collect_by_launch_id(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run

            _save_project_run("run_1", {"launch_id": "launch_1", "project": "demo", "node_name": "n1", "pid": 1, "status": "launched"})
            _save_project_run("run_2", {"launch_id": "launch_1", "project": "demo", "node_name": "n2", "pid": 2, "status": "launched"})

            with patch("crucible.mcp.tools._observe_project_run", side_effect=[
                {"run_id": "run_1", "status": "completed"},
                {"run_id": "run_2", "status": "interrupted"},
            ]):
                result = collect_project_results({"launch_id": "launch_1"})

        assert result["summary"] == {"completed": 1, "interrupted": 1}
        assert {row["run_id"] for row in result["runs"]} == {"run_1", "run_2"}


class TestProjectRunStatus:
    def test_status_includes_recent_events(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run, _append_project_run_event

            _save_project_run("run_123", {"project": "demo", "node_name": "node-1", "pid": 42, "status": "launched"})
            _append_project_run_event("run_123", "launch_requested")
            _append_project_run_event("run_123", "launch_succeeded")

            with patch("crucible.mcp.tools._observe_project_run", return_value={
                "run_id": "run_123",
                "status": "running",
                "launch_id": None,
                "metrics": None,
                "log_tail": "",
                "log_path": "",
                "wandb": None,
                "contract_status": "compliant",
                "failure_class": None,
                "remote_node_state": "ready",
            }):
                result = get_project_run_status({"run_id": "run_123", "event_limit": 2})

        assert result["status"] == "running"
        assert [event["event"] for event in result["events"]] == ["launch_requested", "launch_succeeded"]


class TestProvisionProject:
    def test_uses_next_index_and_interruptible_override(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        _write_spec(
            tmp_path,
            "lewm",
            {
                "name": "lewm",
                "repo": "https://example.com/lewm.git",
                "pod": {
                    "gpu_type": "NVIDIA GeForce RTX 4090",
                    "container_disk": 40,
                    "volume_disk": 80,
                    "interruptible": False,
                },
            },
        )
        (tmp_path / cfg.nodes_file).write_text(
            '[{"name":"lewm-01","node_id":"n1"},{"name":"lewm-04","node_id":"n4"}]',
            encoding="utf-8",
        )
        merged_nodes = [
            {"name": "lewm-01", "node_id": "n1"},
            {"name": "lewm-04", "node_id": "n4"},
            {"name": "lewm-05", "node_id": "n5"},
        ]

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            with patch("crucible.mcp.tools._project_contract_env", return_value={}):
                with patch("crucible.fleet.manager.FleetManager.provision", return_value=merged_nodes) as mock_provision:
                    result = provision_project({"project_name": "lewm", "count": 1})

        assert result["created"] == 1
        assert result["new_nodes"] == [{"name": "lewm-05", "node_id": "n5"}]
        mock_provision.assert_called_once_with(
            count=1,
            name_prefix="lewm",
            start_index=5,
            gpu_type_id="NVIDIA GeForce RTX 4090",
            container_disk_gb=40,
            volume_gb=80,
            interruptible=False,
        )


class TestBootstrapProjectTool:
    def test_includes_node_error_details(self, tmp_path):
        from crucible.core.config import ProjectConfig

        cfg = ProjectConfig(project_root=tmp_path)
        _write_spec(
            tmp_path,
            "demo",
            {
                "name": "demo",
                "repo": "https://example.com/demo.git",
            },
        )
        (tmp_path / cfg.nodes_file).write_text(
            '[{"name":"demo-01","ssh_host":"1.2.3.4","project":"demo"}]',
            encoding="utf-8",
        )

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            with patch(
                "crucible.fleet.bootstrap.bootstrap_project",
                side_effect=RuntimeError("bootstrap:system_tools failed: unsupported_package_manager"),
            ):
                result = bootstrap_project_tool({"project_name": "demo"})

        assert result["bootstrapped"] == 0
        assert result["nodes"] == [
            {
                "name": "demo-01",
                "state": "boot_failed",
                "project": "demo",
                "error": "bootstrap:system_tools failed: unsupported_package_manager",
            }
        ]


class TestRunPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run, _load_project_run

            _save_project_run("run_abc", {
                "node_name": "node-01",
                "ssh_host": "1.2.3.4",
                "project": "myproj",
                "pid": 1234,
            })

            loaded = _load_project_run("run_abc")
            assert loaded is not None
            assert loaded["run_id"] == "run_abc"
            assert loaded["node_name"] == "node-01"
            assert loaded["pid"] == 1234

    def test_load_returns_latest_record(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        (tmp_path / ".crucible" / "projects").mkdir(parents=True, exist_ok=True)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _save_project_run, _load_project_run

            _save_project_run("run_latest", {"status": "launched", "pid": 111})
            _save_project_run("run_latest", {"status": "completed", "pid": 222})

            loaded = _load_project_run("run_latest")
            assert loaded is not None
            assert loaded["status"] == "completed"
            assert loaded["pid"] == 222

    def test_load_missing_returns_none(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _load_project_run
            assert _load_project_run("nonexistent") is None


class TestLaunchIds:
    def test_make_launch_id_is_unique_across_calls(self):
        first = _make_launch_id("lewm")
        second = _make_launch_id("lewm")

        assert first != second
        assert first.startswith("lewm_")
        assert second.startswith("lewm_")
