"""Integration tests for MCP server dispatch and JSON serialization."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import yaml

from crucible.core.config import ProjectConfig
from crucible.core.hub import HubStore
from crucible.mcp.server import call_tool, list_tools


def _decode_response(contents) -> dict:
    assert len(contents) == 1
    return json.loads(contents[0].text)


def test_list_tools_exposes_expected_entries():
    tools = asyncio.run(list_tools())
    names = {tool.name for tool in tools}
    assert "list_projects" in names
    assert "model_list_global_architectures" in names
    assert "model_import_architecture" in names


def test_call_tool_lists_projects_from_server(tmp_path: Path):
    project_root = tmp_path / "project"
    spec_dir = project_root / ".crucible" / "projects"
    spec_dir.mkdir(parents=True)
    (spec_dir / "demo.yaml").write_text(
        yaml.safe_dump({"name": "demo", "repo": "repo/demo", "train": "python train.py"}),
        encoding="utf-8",
    )
    cfg = ProjectConfig(project_root=project_root)

    with patch("crucible.mcp.tools._get_config", return_value=cfg):
        payload = _decode_response(asyncio.run(call_tool("list_projects", {})))

    assert payload["projects"][0]["name"] == "demo"


def test_call_tool_roundtrips_global_spec_listing_and_import(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    hub_dir = tmp_path / "hub"
    hub = HubStore.init(hub_dir=hub_dir, name="hub")
    hub.store_architecture(
        name="yaml_demo",
        code="name: yaml_demo\nbase: tied_embedding_lm\nembedding: {}\nblock: {}\nstack: {}\n",
        kind="spec",
    )
    cfg = ProjectConfig(project_root=project_root, hub_dir=str(hub_dir))

    with patch("crucible.mcp.tools._get_config", return_value=cfg):
        listed = _decode_response(asyncio.run(call_tool("model_list_global_architectures", {})))
        imported = _decode_response(asyncio.run(call_tool("model_import_architecture", {"name": "yaml_demo"})))

    assert any(
        architecture["name"] == "yaml_demo" and architecture["kind"] == "spec"
        for architecture in listed["architectures"]
    )
    assert imported["status"] == "imported"
    assert imported["family"] == "yaml_demo"
    assert (project_root / ".crucible" / "architectures" / "yaml_demo.yaml").exists()


def test_call_tool_unknown_name_returns_json_error():
    payload = _decode_response(asyncio.run(call_tool("does_not_exist", {})))
    assert "Unknown tool" in payload["error"]
