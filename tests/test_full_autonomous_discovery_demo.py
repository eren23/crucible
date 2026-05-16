"""Smoke test for the Phase 4.4 end-to-end demo project.

Verifies that the playbook in
``examples/full_autonomous_discovery/README.md`` lines up with the
shipped MCP surface — every command the README references is a valid
TOOL_DISPATCH entry, and the canned project spec loads cleanly.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


DEMO_DIR = Path(__file__).parent.parent / "examples" / "full_autonomous_discovery"


class TestDemoProjectStructure:
    def test_demo_dir_exists(self):
        assert DEMO_DIR.exists()
        assert (DEMO_DIR / "crucible.yaml").exists()
        assert (DEMO_DIR / "train.py").exists()
        assert (DEMO_DIR / "README.md").exists()

    def test_crucible_yaml_loads(self, tmp_path, monkeypatch):
        # Copy to a clean dir so we don't disturb the in-repo demo.
        target = tmp_path / "demo"
        shutil.copytree(DEMO_DIR, target)
        monkeypatch.chdir(target)
        from crucible.core.config import load_config
        config = load_config()
        assert config.name == "full-autonomous-discovery"
        assert config.metrics.primary == "val_loss"


class TestPlaybookToolsExist:
    """Every MCP tool the README references must be registered."""

    @pytest.mark.parametrize("tool_name", [
        "tool_router",
        "research_arxiv_search",
        "autonomous_research_loop",
        "design_synthesize_from_findings",
        "runs_search",
        "get_research_briefing",
        "note_generate_paper_draft",
        "research_peer_sync",
    ])
    def test_tool_in_dispatch(self, tool_name):
        from crucible.mcp.tools import TOOL_DISPATCH
        assert tool_name in TOOL_DISPATCH, (
            f"Demo README references {tool_name!r} but it's not in TOOL_DISPATCH"
        )


class TestPlaybookCommandsLineUp:
    """The README's CLI commands must resolve to actual entry points."""

    def test_cli_subcommands_registered(self):
        """`crucible {init,run,analyze,recipe,mcp}` all exist —
        the README invokes each at some point."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "crucible.cli.main", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        # `crucible --help` lists every registered top-level command.
        out = result.stdout + result.stderr
        for cmd in ("init", "run", "analyze", "recipe", "mcp"):
            assert cmd in out, (
                f"Demo README references `crucible {cmd}` but it's not in --help"
            )


class TestMemoryFilterPolicyExposed:
    """Phase 4.2 memory_filter must be selectable via the
    design_synthesize_from_findings MCP."""

    def test_memory_filter_in_synthesis_policy_enum(self):
        # Read the registered Tool() schema for design_synthesize_from_findings
        # and assert the enum includes our new policy.
        from crucible.mcp.server import TOOLS
        for tool in TOOLS:
            if tool.name == "design_synthesize_from_findings":
                policy_prop = (
                    tool.inputSchema.get("properties", {}).get("policy", {})
                )
                assert "memory_filter" in policy_prop.get("enum", []), (
                    "Phase 4.2 memory_filter policy must be in the schema enum"
                )
                return
        pytest.fail("design_synthesize_from_findings tool not registered")
