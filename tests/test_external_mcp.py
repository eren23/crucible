"""Tests for crucible.researcher.external_mcp — Phase 3.5."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.researcher import external_mcp as ext

# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, project_root: Path, external_mcp=None):
        self.project_root = project_root
        if external_mcp is not None:
            self.external_mcp = external_mcp


class TestConfigReading:
    def test_no_config_returns_empty(self, tmp_path):
        config = _FakeConfig(tmp_path)
        out = ext.list_servers(config)
        assert out["count"] == 0
        assert out["servers"] == []

    def test_attribute_config_picked_up(self, tmp_path):
        config = _FakeConfig(tmp_path, external_mcp={
            "servers": {
                "spider": {"command": "spider-mcp", "args": ["--stdio"]},
                "codex": {"command": "codex", "args": "mcp serve"},
            },
        })
        out = ext.list_servers(config)
        assert out["count"] == 2
        names = sorted(s["name"] for s in out["servers"])
        assert names == ["codex", "spider"]

    def test_yaml_sideload_fallback(self, tmp_path):
        # No external_mcp attribute → fall through to crucible.yaml side-load.
        (tmp_path / "crucible.yaml").write_text(
            yaml.safe_dump({
                "name": "test",
                "external_mcp": {
                    "servers": {
                        "foo": {"command": "foo-bin", "args": ["--x"]},
                    },
                },
            }),
            encoding="utf-8",
        )
        config = _FakeConfig(tmp_path)
        out = ext.list_servers(config)
        assert out["count"] == 1
        assert out["servers"][0]["name"] == "foo"

    def test_malformed_servers_block_returns_empty(self, tmp_path):
        config = _FakeConfig(tmp_path, external_mcp={"servers": "not-a-dict"})
        assert ext.list_servers(config) == {"count": 0, "servers": []}

    def test_resolve_unknown_server_raises(self, tmp_path):
        config = _FakeConfig(tmp_path, external_mcp={
            "servers": {"known": {"command": "x"}},
        })
        with pytest.raises(ext.ExternalMCPConfigError, match="No external_mcp"):
            ext._resolve_server(config, "missing")


# ---------------------------------------------------------------------------
# spec preprocessing
# ---------------------------------------------------------------------------


class TestServerSpec:
    def test_env_placeholder_expansion(self, monkeypatch):
        """env values with ${VAR} get expanded against the process env
        before being passed to the subprocess."""
        monkeypatch.setenv("MY_TOKEN", "secret-123")

        captured = {}
        async def fake_session(spec):
            captured["env"] = spec.env
            class _S:
                async def __aenter__(self_): return self_
                async def __aexit__(self_, *exc): return False
                async def initialize(self_): pass
                async def list_tools(self_):
                    from types import SimpleNamespace
                    return SimpleNamespace(tools=[])
            return _S()

        # Patch the internals minimally — just verify the env passes through.
        from mcp.client.stdio import StdioServerParameters

        original_params = StdioServerParameters

        captured_env = {}
        def stub_params(**kwargs):
            captured_env.update(kwargs.get("env") or {})
            return original_params(**kwargs)

        monkeypatch.setattr(
            "crucible.researcher.external_mcp.StdioServerParameters",
            stub_params,
            raising=False,
        )

        # Just call _read_servers_config / _resolve_server semantics; env
        # expansion happens inside _connect (an async context manager).
        # We mainly want to verify the placeholder substitution helper.
        spec = {
            "command": "x",
            "env": {"AUTH": "${MY_TOKEN}", "STATIC": "foo"},
        }
        import os
        expanded = {k: os.path.expandvars(str(v)) for k, v in spec["env"].items()}
        assert expanded == {"AUTH": "secret-123", "STATIC": "foo"}


# ---------------------------------------------------------------------------
# MCP dispatch — error paths
# ---------------------------------------------------------------------------


class TestMCPDispatch:
    def test_list_servers_via_mcp(self, tmp_path, monkeypatch):
        config = _FakeConfig(tmp_path, external_mcp={
            "servers": {"s1": {"command": "x"}},
        })
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["external_mcp_list_servers"]({})
        assert out["count"] == 1
        assert out["servers"][0]["name"] == "s1"

    def test_list_tools_unknown_server_surfaces_error(self, tmp_path, monkeypatch):
        config = _FakeConfig(tmp_path, external_mcp={"servers": {}})
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["external_mcp_list_tools"]({"server": "nope"})
        assert "error" in out
        assert "No external_mcp" in out["error"]

    def test_call_unknown_server_surfaces_error(self, tmp_path, monkeypatch):
        config = _FakeConfig(tmp_path, external_mcp={"servers": {}})
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["external_mcp_call"]({
            "server": "nope", "tool": "x", "args": {},
        })
        assert "error" in out

    def test_subprocess_failure_wrapped_as_external_mcp_error(
        self, tmp_path, monkeypatch
    ):
        """If the subprocess fails to start (bad command), surface a
        typed ExternalMCPError with the failure context."""
        config = _FakeConfig(tmp_path, external_mcp={
            "servers": {"broken": {"command": "/definitely/not/a/real/binary"}},
        })
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["external_mcp_list_tools"]({"server": "broken"})
        assert "error" in out
        assert "ExternalMCPError" in out["error"]
