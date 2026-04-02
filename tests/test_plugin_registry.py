"""Tests for the generic PluginRegistry and plugin discovery."""
from __future__ import annotations

import pytest
from pathlib import Path

from crucible.core.errors import PluginError
from crucible.core.plugin_registry import PluginRegistry
from crucible.core.plugin_discovery import discover_all_plugins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry():
    r = PluginRegistry[str]("test_plugin")
    yield r
    r.reset()


# ---------------------------------------------------------------------------
# Registration basics
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_and_build(self, registry: PluginRegistry):
        registry.register("echo", lambda msg="hi": msg, source="builtin")
        assert registry.build("echo") == "hi"
        assert registry.build("echo", msg="hello") == "hello"

    def test_register_appears_in_list(self, registry: PluginRegistry):
        registry.register("alpha", lambda: "a", source="builtin")
        registry.register("beta", lambda: "b", source="builtin")
        assert registry.list_plugins() == ["alpha", "beta"]

    def test_build_unknown_raises(self, registry: PluginRegistry):
        with pytest.raises(PluginError, match="Unknown test_plugin"):
            registry.build("nonexistent")

    def test_contains(self, registry: PluginRegistry):
        registry.register("x", lambda: 1, source="builtin")
        assert "x" in registry
        assert "y" not in registry

    def test_len(self, registry: PluginRegistry):
        assert len(registry) == 0
        registry.register("a", lambda: 1, source="builtin")
        assert len(registry) == 1


class TestDescribePlugin:
    def test_describe_plugin_returns_metadata(self, registry: PluginRegistry):
        class EchoFactory:
            pass

        registry.register("echo", EchoFactory, source="global")
        info = registry.describe_plugin("echo")
        assert info == {"name": "echo", "type": "EchoFactory", "source": "global"}

    def test_describe_plugin_unknown_is_none(self, registry: PluginRegistry):
        assert registry.describe_plugin("missing") is None


# ---------------------------------------------------------------------------
# 3-tier precedence
# ---------------------------------------------------------------------------

class TestPrecedence:
    def test_higher_source_overrides_lower(self, registry: PluginRegistry):
        registry.register("opt", lambda: "builtin", source="builtin")
        registry.register("opt", lambda: "global", source="global")
        assert registry.build("opt") == "global"

    def test_local_overrides_global(self, registry: PluginRegistry):
        registry.register("opt", lambda: "global", source="global")
        registry.register("opt", lambda: "local", source="local")
        assert registry.build("opt") == "local"

    def test_local_overrides_builtin(self, registry: PluginRegistry):
        registry.register("opt", lambda: "builtin", source="builtin")
        registry.register("opt", lambda: "local", source="local")
        assert registry.build("opt") == "local"

    def test_equal_precedence_raises(self, registry: PluginRegistry):
        registry.register("opt", lambda: "first", source="global")
        with pytest.raises(PluginError, match="already registered"):
            registry.register("opt", lambda: "second", source="global")

    def test_lower_precedence_raises(self, registry: PluginRegistry):
        registry.register("opt", lambda: "local", source="local")
        with pytest.raises(PluginError, match="already registered"):
            registry.register("opt", lambda: "builtin", source="builtin")

    def test_invalid_source_raises(self, registry: PluginRegistry):
        with pytest.raises(PluginError, match="Invalid source"):
            registry.register("opt", lambda: "x", source="typo")

    def test_detailed_list_includes_source(self, registry: PluginRegistry):
        registry.register("a", lambda: 1, source="builtin")
        registry.register("b", lambda: 2, source="local")
        detailed = registry.list_plugins_detailed()
        sources = {d["name"]: d["source"] for d in detailed}
        assert sources == {"a": "builtin", "b": "local"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_register_and_get_schema(self, registry: PluginRegistry):
        schema = {"lr": {"type": "float", "default": 0.001}}
        registry.register_schema("opt", schema)
        assert registry.get_schema("opt") == schema

    def test_missing_schema_returns_stub(self, registry: PluginRegistry):
        result = registry.get_schema("missing")
        assert "No schema" in result["note"]


# ---------------------------------------------------------------------------
# File-based discovery
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    def test_load_plugins_from_directory(self, registry: PluginRegistry, tmp_path: Path):
        plugin_code = (
            "from crucible.core.plugin_registry import PluginRegistry\n"
            "# The test injects the registry via module-level exec, but in real\n"
            "# usage the plugin imports the specific registry instance.\n"
        )
        (tmp_path / "my_plugin.py").write_text(plugin_code)

        # Simply verify it loads without error (the plugin doesn't register
        # anything, but the discovery mechanism runs)
        loaded = registry.load_plugins(tmp_path, source="local")
        assert loaded == ["my_plugin"]

    def test_skips_underscore_files(self, registry: PluginRegistry, tmp_path: Path):
        (tmp_path / "_private.py").write_text("x = 1")
        (tmp_path / "__init__.py").write_text("")
        loaded = registry.load_plugins(tmp_path, source="local")
        assert loaded == []

    def test_skips_broken_plugins(self, registry: PluginRegistry, tmp_path: Path):
        (tmp_path / "broken.py").write_text("raise RuntimeError('boom')")
        loaded = registry.load_plugins(tmp_path, source="local")
        # Broken plugin is skipped silently
        assert loaded == []

    def test_missing_directory_returns_empty(self, registry: PluginRegistry, tmp_path: Path):
        loaded = registry.load_plugins(tmp_path / "nonexistent", source="local")
        assert loaded == []


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_everything(self, registry: PluginRegistry):
        registry.register("x", lambda: 1, source="builtin")
        registry.register_schema("x", {"a": 1})
        registry.reset()
        assert len(registry) == 0
        assert registry.list_plugins() == []
        assert "No schema" in registry.get_schema("x")["note"]


# ---------------------------------------------------------------------------
# discover_all_plugins
# ---------------------------------------------------------------------------

class TestDiscoverAll:
    def test_discovers_from_local_dir(self, tmp_path: Path):
        reg = PluginRegistry[str]("test")
        try:
            plugins_dir = tmp_path / ".crucible" / "plugins" / "test_type"
            plugins_dir.mkdir(parents=True)
            (plugins_dir / "foo.py").write_text("x = 1")

            result = discover_all_plugins(
                {"test_type": reg},
                project_root=tmp_path,
                hub_dir=tmp_path / "fake_hub",
            )
            assert "foo" in result["test_type"]
        finally:
            reg.reset()

    def test_no_project_root_skips_local(self, tmp_path: Path):
        reg = PluginRegistry[str]("test")
        try:
            result = discover_all_plugins(
                {"test_type": reg},
                project_root=None,
                hub_dir=tmp_path / "fake_hub",
            )
            assert result["test_type"] == []
        finally:
            reg.reset()
