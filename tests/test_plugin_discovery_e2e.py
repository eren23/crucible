"""End-to-end tests for plugin discovery and registration.

Tests the full cycle: write plugin file -> discover -> registry populated ->
build works.  Covers optimizers, schedulers, providers, callbacks, loggers.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from crucible.core.plugin_registry import PluginRegistry
from crucible.core.plugin_discovery import discover_all_plugins
from crucible.core.errors import PluginError


class TestE2EOptimizerPlugin:
    """Write an optimizer plugin, discover it, use it."""

    def test_discover_and_build_custom_optimizer(self, tmp_path: Path):
        from crucible.training.optimizers import OPTIMIZER_REGISTRY, build_optimizer

        plugin_dir = tmp_path / ".crucible" / "plugins" / "optimizers"
        plugin_dir.mkdir(parents=True)

        code = textwrap.dedent("""\
            from crucible.training.optimizers import register_optimizer

            def _lion_factory(params, **kwargs):
                return {"type": "lion", "params_count": len(params) if hasattr(params, '__len__') else 0, "kwargs": kwargs}

            register_optimizer("test_lion", _lion_factory, source="local")
        """)
        (plugin_dir / "lion.py").write_text(code)

        try:
            result = discover_all_plugins(
                {"optimizers": OPTIMIZER_REGISTRY},
                project_root=tmp_path,
                hub_dir=tmp_path / "fake_hub",
            )
            assert "lion" in result["optimizers"]
            assert "test_lion" in OPTIMIZER_REGISTRY.list_plugins()

            # Actually build it
            opt = build_optimizer("test_lion", [1, 2, 3], lr=0.001)
            assert opt["type"] == "lion"
            assert opt["kwargs"]["lr"] == 0.001
        finally:
            OPTIMIZER_REGISTRY.unregister("test_lion")
            sys.modules.pop("_crucible_plugin_optimizer_local_lion", None)

    def test_local_overrides_global(self, tmp_path: Path):
        """A local plugin should override a global one with the same name."""
        from crucible.training.optimizers import OPTIMIZER_REGISTRY

        hub = tmp_path / "hub"
        global_dir = hub / "plugins" / "optimizers"
        global_dir.mkdir(parents=True)
        local_dir = tmp_path / "proj" / ".crucible" / "plugins" / "optimizers"
        local_dir.mkdir(parents=True)

        (global_dir / "clash.py").write_text(textwrap.dedent("""\
            from crucible.training.optimizers import register_optimizer
            register_optimizer("test_clash", lambda p, **kw: "global_version", source="global")
        """))
        (local_dir / "clash.py").write_text(textwrap.dedent("""\
            from crucible.training.optimizers import register_optimizer
            register_optimizer("test_clash", lambda p, **kw: "local_version", source="local")
        """))

        try:
            discover_all_plugins(
                {"optimizers": OPTIMIZER_REGISTRY},
                project_root=tmp_path / "proj",
                hub_dir=hub,
            )
            from crucible.training.optimizers import build_optimizer
            result = build_optimizer("test_clash", [])
            assert result == "local_version"
        finally:
            OPTIMIZER_REGISTRY.unregister("test_clash")
            sys.modules.pop("_crucible_plugin_optimizer_global_clash", None)
            sys.modules.pop("_crucible_plugin_optimizer_local_clash", None)


class TestE2ESchedulerPlugin:
    """Write a scheduler plugin, discover it, use it."""

    def test_discover_and_build_custom_scheduler(self, tmp_path: Path):
        from crucible.training.schedulers import SCHEDULER_REGISTRY, build_scheduler

        plugin_dir = tmp_path / ".crucible" / "plugins" / "schedulers"
        plugin_dir.mkdir(parents=True)

        code = textwrap.dedent("""\
            from crucible.training.schedulers import register_scheduler

            def _step_factory(optimizer, *, total_steps=0, warmup_steps=0, step_size=10, **kwargs):
                return {"type": "step_decay", "step_size": step_size}

            register_scheduler("test_step_decay", _step_factory, source="local")
        """)
        (plugin_dir / "step_decay.py").write_text(code)

        try:
            discover_all_plugins(
                {"schedulers": SCHEDULER_REGISTRY},
                project_root=tmp_path,
                hub_dir=tmp_path / "fake_hub",
            )
            sched = build_scheduler("test_step_decay", object(), step_size=30)
            assert sched["type"] == "step_decay"
            assert sched["step_size"] == 30
        finally:
            SCHEDULER_REGISTRY.unregister("test_step_decay")
            sys.modules.pop("_crucible_plugin_scheduler_local_step_decay", None)


class TestE2ECallbackPlugin:
    def test_discover_and_build_custom_callback(self, tmp_path: Path):
        from crucible.training.callbacks import CALLBACK_REGISTRY, build_callback

        plugin_dir = tmp_path / ".crucible" / "plugins" / "callbacks"
        plugin_dir.mkdir(parents=True)

        code = textwrap.dedent("""\
            from crucible.training.callbacks import TrainingCallback, register_callback

            class CheckpointCallback(TrainingCallback):
                priority = 50
                def __init__(self, **kwargs):
                    self.save_every = kwargs.get("save_every", 100)
                def on_step_end(self, step, metrics, state):
                    if step % self.save_every == 0:
                        state.setdefault("checkpoints", []).append(step)

            register_callback("test_checkpoint", CheckpointCallback, source="local")
        """)
        (plugin_dir / "checkpoint.py").write_text(code)

        try:
            discover_all_plugins(
                {"callbacks": CALLBACK_REGISTRY},
                project_root=tmp_path,
                hub_dir=tmp_path / "fake_hub",
            )
            cb = build_callback("test_checkpoint", save_every=50)
            assert cb.priority == 50
            assert cb.save_every == 50

            # Actually fire the callback
            state = {}
            cb.on_step_end(100, {"loss": 0.1}, state)
            assert state["checkpoints"] == [100]
        finally:
            CALLBACK_REGISTRY.unregister("test_checkpoint")
            sys.modules.pop("_crucible_plugin_callback_local_checkpoint", None)


class TestE2ELoggerPlugin:
    def test_discover_and_build_custom_logger(self, tmp_path: Path):
        from crucible.runner.loggers import LOGGER_REGISTRY, build_logger

        plugin_dir = tmp_path / ".crucible" / "plugins" / "loggers"
        plugin_dir.mkdir(parents=True)

        code = textwrap.dedent("""\
            from crucible.runner.loggers import TrainingLogger, register_logger

            class InMemoryLogger(TrainingLogger):
                entries = []
                def __init__(self, **kwargs):
                    self.entries = []
                def log(self, metrics, *, step=None):
                    self.entries.append({"step": step, **metrics})
                def finish(self, exit_code=0):
                    pass

            register_logger("test_inmemory", InMemoryLogger, source="local")
        """)
        (plugin_dir / "inmemory.py").write_text(code)

        try:
            discover_all_plugins(
                {"loggers": LOGGER_REGISTRY},
                project_root=tmp_path,
                hub_dir=tmp_path / "fake_hub",
            )
            logger = build_logger("test_inmemory")
            logger.log({"loss": 0.5}, step=1)
            logger.log({"loss": 0.3}, step=2)
            assert len(logger.entries) == 2
            assert logger.entries[0]["loss"] == 0.5
        finally:
            LOGGER_REGISTRY.unregister("test_inmemory")
            sys.modules.pop("_crucible_plugin_logger_local_inmemory", None)


class TestDiscoverMultipleRegistries:
    """discover_all_plugins with multiple registries at once."""

    def test_discovers_across_multiple_types(self, tmp_path: Path):
        r1 = PluginRegistry("type_a")
        r2 = PluginRegistry("type_b")

        dir_a = tmp_path / ".crucible" / "plugins" / "type_a"
        dir_a.mkdir(parents=True)
        dir_b = tmp_path / ".crucible" / "plugins" / "type_b"
        dir_b.mkdir(parents=True)

        (dir_a / "plug_a.py").write_text("x = 'hello_a'\n")
        (dir_b / "plug_b.py").write_text("x = 'hello_b'\n")

        try:
            result = discover_all_plugins(
                {"type_a": r1, "type_b": r2},
                project_root=tmp_path,
                hub_dir=tmp_path / "fake_hub",
            )
            assert "plug_a" in result["type_a"]
            assert "plug_b" in result["type_b"]
        finally:
            r1.reset()
            r2.reset()
            sys.modules.pop("_crucible_plugin_type_a_local_plug_a", None)
            sys.modules.pop("_crucible_plugin_type_b_local_plug_b", None)

    def test_empty_dirs_yield_empty_results(self, tmp_path: Path):
        r = PluginRegistry("empty_type")
        result = discover_all_plugins(
            {"empty_type": r},
            project_root=tmp_path,
            hub_dir=tmp_path / "nonexistent_hub",
        )
        assert result["empty_type"] == []
        r.reset()
