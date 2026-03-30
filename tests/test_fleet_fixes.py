"""Tests for fleet infrastructure fixes — scp_to_node and ProjectSpec.local_files."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from crucible.fleet.sync import scp_to_node
from crucible.core.config import ProjectSpec, load_project_spec


def _make_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "test-node",
        "ssh_host": "10.0.0.1",
        "ssh_port": 2222,
        "user": "ubuntu",
        "ssh_key": "~/.ssh/id_ed25519",
        "workspace_path": "/workspace/project",
        "connect_timeout": 12,
    }
    base.update(overrides)
    return base


def _write_spec(tmp_path: Path, name: str, content: dict[str, Any]) -> Path:
    d = tmp_path / ".crucible" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(content), encoding="utf-8")
    return p


class TestScpToNode:
    @patch("crucible.fleet.sync._run")
    def test_scp_to_node_command(self, mock_run):
        """Verify scp command is correctly built with host, port, and key."""
        node = _make_node()
        scp_to_node(node, "/tmp/local_file.py", "/workspace/project/file.py")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "scp"
        assert "-P" in cmd
        p_idx = cmd.index("-P")
        assert cmd[p_idx + 1] == "2222"
        assert "-i" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        assert "UserKnownHostsFile=/dev/null" in cmd
        assert "BatchMode=yes" in cmd
        assert "/tmp/local_file.py" in cmd
        assert "ubuntu@10.0.0.1:/workspace/project/file.py" in cmd


class TestProjectSpecLocalFiles:
    def test_project_spec_local_files(self, tmp_path: Path):
        """Load a YAML with local_files and verify it parses."""
        _write_spec(tmp_path, "with_files", {
            "name": "with_files",
            "repo": "https://github.com/user/repo.git",
            "local_files": ["scripts/foo.py", "configs/bar.yaml"],
        })
        spec = load_project_spec("with_files", tmp_path)
        assert spec.local_files == ["scripts/foo.py", "configs/bar.yaml"]

    def test_project_spec_local_files_default_empty(self, tmp_path: Path):
        """Verify default local_files is empty list when not specified."""
        _write_spec(tmp_path, "no_files", {
            "name": "no_files",
            "repo": "https://github.com/user/repo.git",
        })
        spec = load_project_spec("no_files", tmp_path)
        assert spec.local_files == []

    def test_project_spec_dataclass_default(self):
        """Verify the dataclass default for local_files is empty list."""
        spec = ProjectSpec(name="test", repo="https://example.com/r.git")
        assert spec.local_files == []
