"""End-to-end MCP stdio protocol tests against a real server subprocess."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from crucible.core.hub import HubStore


def _decode_call_result(result) -> dict:
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


async def _run_stdio_session(
    project_root: Path,
    callback,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(src_root)
        if not existing_pythonpath
        else os.pathsep.join([str(src_root), existing_pythonpath])
    )

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "crucible.mcp.server"],
        cwd=project_root,
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await callback(session)


def test_stdio_server_lists_tools_and_projects(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "crucible.yaml").write_text("name: stdio-project\n", encoding="utf-8")

    specs_dir = project_root / ".crucible" / "projects"
    specs_dir.mkdir(parents=True)
    (specs_dir / "demo.yaml").write_text(
        yaml.safe_dump({"name": "demo", "repo": "repo/demo", "train": "python train.py"}),
        encoding="utf-8",
    )

    async def scenario(session: ClientSession) -> None:
        tools = await session.list_tools()
        tool_names = {tool.name for tool in tools.tools}
        assert "list_projects" in tool_names
        assert "model_list_global_architectures" in tool_names

        result = await session.call_tool("list_projects", {})
        payload = _decode_call_result(result)
        assert payload["projects"][0]["name"] == "demo"

    asyncio.run(_run_stdio_session(project_root, scenario))


def test_stdio_server_roundtrips_global_spec_import(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    hub_dir = tmp_path / "hub"
    hub = HubStore.init(hub_dir=hub_dir, name="hub")
    hub.store_architecture(
        name="yaml_demo",
        code="name: yaml_demo\nbase: tied_embedding_lm\nembedding: {}\nblock: {}\nstack: {}\n",
        kind="spec",
    )
    (project_root / "crucible.yaml").write_text(
        yaml.safe_dump({"name": "stdio-project", "hub_dir": str(hub_dir)}),
        encoding="utf-8",
    )

    async def scenario(session: ClientSession) -> None:
        listed = await session.call_tool("model_list_global_architectures", {})
        listed_payload = _decode_call_result(listed)
        assert any(
            architecture["name"] == "yaml_demo" and architecture["kind"] == "spec"
            for architecture in listed_payload["architectures"]
        )

        imported = await session.call_tool("model_import_architecture", {"name": "yaml_demo"})
        imported_payload = _decode_call_result(imported)
        assert imported_payload["status"] == "imported"
        assert imported_payload["family"] == "yaml_demo"

    asyncio.run(_run_stdio_session(project_root, scenario))

    assert (project_root / ".crucible" / "architectures" / "yaml_demo.yaml").exists()
