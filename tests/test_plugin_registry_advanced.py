"""Advanced PluginRegistry tests — threading, edge cases, get(), unregister.

Supplements test_plugin_registry.py with deeper coverage.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from crucible.core.errors import PluginError
from crucible.core.plugin_registry import PluginRegistry


class TestGet:
    def test_get_returns_factory_when_registered(self):
        r = PluginRegistry("test_get")
        f = lambda: 42
        r.register("x", f, source="builtin")
        assert r.get("x") is f
        r.reset()

    def test_get_returns_none_when_missing(self):
        r = PluginRegistry("test_get2")
        assert r.get("nonexistent") is None
        r.reset()

    def test_get_after_override_returns_latest(self):
        r = PluginRegistry("test_get3")
        r.register("x", lambda: "v1", source="builtin")
        r.register("x", lambda: "v2", source="local")
        assert r.get("x")() == "v2"
        r.reset()


class TestBuildKwargsForwarding:
    def test_kwargs_forwarded_to_factory(self):
        r = PluginRegistry("test_kwargs")

        def factory(*, a, b, c=10):
            return {"a": a, "b": b, "c": c}

        r.register("fac", factory, source="builtin")
        result = r.build("fac", a=1, b=2, c=3)
        assert result == {"a": 1, "b": 2, "c": 3}
        r.reset()

    def test_build_with_no_kwargs(self):
        r = PluginRegistry("test_no_kwargs")
        r.register("simple", lambda: "hello", source="builtin")
        assert r.build("simple") == "hello"
        r.reset()


class TestConcurrentAccess:
    """Basic thread safety: concurrent register + build shouldn't crash."""

    def test_concurrent_register_and_build(self):
        r = PluginRegistry("test_concurrent")
        errors = []
        barrier = threading.Barrier(4)

        def register_worker(name, value):
            try:
                barrier.wait(timeout=5)
                r.register(name, lambda v=value: v, source="builtin")
            except Exception as e:
                errors.append(e)

        def build_worker():
            try:
                barrier.wait(timeout=5)
                # May or may not find "a" depending on timing — should never crash
                time.sleep(0.01)
                try:
                    r.build("a")
                except PluginError:
                    pass  # Expected if "a" not registered yet
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_worker, args=("a", 1)),
            threading.Thread(target=register_worker, args=("b", 2)),
            threading.Thread(target=register_worker, args=("c", 3)),
            threading.Thread(target=build_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(r) == 3
        r.reset()

    def test_concurrent_unregister(self):
        r = PluginRegistry("test_conc_unreg")
        for i in range(10):
            r.register(f"item_{i}", lambda i=i: i, source="builtin")

        errors = []

        def unreg_worker(name):
            try:
                r.unregister(name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=unreg_worker, args=(f"item_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(r) == 0
        r.reset()


class TestPrecedenceEdgeCases:
    def test_full_override_chain(self):
        """builtin -> global -> local should work in sequence."""
        r = PluginRegistry("test_chain")
        r.register("x", lambda: "builtin", source="builtin")
        assert r.build("x") == "builtin"

        r.register("x", lambda: "global", source="global")
        assert r.build("x") == "global"

        r.register("x", lambda: "local", source="local")
        assert r.build("x") == "local"
        r.reset()

    def test_global_cannot_override_local(self):
        r = PluginRegistry("test_no_downgrade")
        r.register("x", lambda: "local", source="local")
        with pytest.raises(PluginError, match="already registered"):
            r.register("x", lambda: "global", source="global")
        r.reset()

    def test_builtin_cannot_override_global(self):
        r = PluginRegistry("test_no_downgrade2")
        r.register("x", lambda: "global", source="global")
        with pytest.raises(PluginError, match="already registered"):
            r.register("x", lambda: "builtin", source="builtin")
        r.reset()

    def test_unregister_then_reregister_at_lower_precedence(self):
        """After unregister, any precedence should work."""
        r = PluginRegistry("test_unreg_rereg")
        r.register("x", lambda: "local", source="local")
        r.unregister("x")
        # Now builtin should work
        r.register("x", lambda: "builtin", source="builtin")
        assert r.build("x") == "builtin"
        r.reset()


class TestLoadPluginsEdgeCases:
    def test_empty_directory(self, tmp_path: Path):
        r = PluginRegistry("test_empty")
        loaded = r.load_plugins(tmp_path, source="local")
        assert loaded == []
        r.reset()

    def test_directory_with_only_init(self, tmp_path: Path):
        (tmp_path / "__init__.py").write_text("")
        r = PluginRegistry("test_init")
        loaded = r.load_plugins(tmp_path, source="local")
        assert loaded == []
        r.reset()

    def test_directory_with_only_underscore_files(self, tmp_path: Path):
        (tmp_path / "_private.py").write_text("x = 1")
        (tmp_path / "__pycache__").mkdir()
        r = PluginRegistry("test_underscore")
        loaded = r.load_plugins(tmp_path, source="local")
        assert loaded == []
        r.reset()

    def test_multiple_plugins_loaded_in_sort_order(self, tmp_path: Path):
        (tmp_path / "zebra.py").write_text("x = 1")
        (tmp_path / "alpha.py").write_text("x = 2")
        (tmp_path / "middle.py").write_text("x = 3")
        r = PluginRegistry("test_sort")
        loaded = r.load_plugins(tmp_path, source="local")
        assert loaded == ["alpha", "middle", "zebra"]
        r.reset()

    def test_syntax_error_plugin_skipped(self, tmp_path: Path):
        (tmp_path / "good.py").write_text("x = 1")
        (tmp_path / "bad.py").write_text("def broken(:\n")
        r = PluginRegistry("test_syntax")
        loaded = r.load_plugins(tmp_path, source="local")
        assert "good" in loaded
        assert "bad" not in loaded
        r.reset()


class TestResetIsolation:
    def test_reset_then_register_works(self):
        r = PluginRegistry("test_reset2")
        r.register("a", lambda: 1, source="builtin")
        r.reset()
        assert len(r) == 0
        # Should work after reset
        r.register("a", lambda: 2, source="builtin")
        assert r.build("a") == 2
        r.reset()

    def test_reset_does_not_affect_other_registries(self):
        r1 = PluginRegistry("test_iso1")
        r2 = PluginRegistry("test_iso2")
        r1.register("x", lambda: 1, source="builtin")
        r2.register("x", lambda: 2, source="builtin")
        r1.reset()
        assert len(r1) == 0
        assert len(r2) == 1
        assert r2.build("x") == 2
        r2.reset()
