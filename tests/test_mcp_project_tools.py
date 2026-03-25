"""Tests for MCP external project tools."""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from crucible.mcp.tools import list_projects, collect_project_results


def _write_spec(tmp_path, name, content):
    d = tmp_path / ".crucible" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(content), encoding="utf-8")


class TestListProjects:
    def test_lists_specs(self, tmp_path, monkeypatch):
        _write_spec(tmp_path, "proj1", {"name": "proj1", "repo": "r1", "train": "t1"})
        _write_spec(tmp_path, "proj2", {"name": "proj2", "repo": "r2", "train": "t2"})

        # Mock _get_config to return our tmp_path
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)
        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            result = list_projects({})

        assert len(result["projects"]) == 2
        names = {p["name"] for p in result["projects"]}
        assert names == {"proj1", "proj2"}

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

    def test_load_missing_returns_none(self, tmp_path):
        from crucible.core.config import ProjectConfig
        cfg = ProjectConfig(project_root=tmp_path)

        with patch("crucible.mcp.tools._get_config", return_value=cfg):
            from crucible.mcp.tools import _load_project_run
            assert _load_project_run("nonexistent") is None
