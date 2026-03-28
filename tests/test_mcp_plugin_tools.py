"""Tests for the 15 new MCP plugin registry tools.

Exercises the tool handlers directly (not via MCP transport).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import ProjectConfig


@pytest.fixture()
def mcp_config(tmp_path: Path, monkeypatch):
    """Provide a fake config pointing to a tmp project directory."""
    project = tmp_path / "proj"
    project.mkdir()
    config = ProjectConfig()
    config.project_root = project
    config.store_dir = ".crucible"
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: config)
    return config


# ===================================================================
# List tools — verify return shape and content
# ===================================================================

class TestListTools:
    def test_optimizer_list_available(self):
        from crucible.mcp.tools import optimizer_list_available
        result = optimizer_list_available({})
        assert "optimizers" in result
        names = [d["name"] for d in result["optimizers"]]
        assert "adam" in names
        assert "adamw" in names
        assert "muon" in names
        assert "sgd" in names
        assert "rmsprop" in names
        # Check shape
        for entry in result["optimizers"]:
            assert "name" in entry
            assert "source" in entry

    def test_scheduler_list_available(self):
        from crucible.mcp.tools import scheduler_list_available
        result = scheduler_list_available({})
        assert "schedulers" in result
        names = [d["name"] for d in result["schedulers"]]
        assert "cosine" in names
        assert "constant" in names
        assert "linear" in names
        assert "cosine_restarts" in names

    def test_provider_list_available(self):
        from crucible.mcp.tools import provider_list_available
        result = provider_list_available({})
        assert "providers" in result
        names = [d["name"] for d in result["providers"]]
        assert "runpod" in names
        assert "ssh" in names

    def test_logger_list_available(self):
        from crucible.mcp.tools import logger_list_available
        result = logger_list_available({})
        assert "loggers" in result
        names = [d["name"] for d in result["loggers"]]
        assert "console" in names
        assert "jsonl" in names
        assert "wandb" in names

    def test_callback_list_available(self):
        from crucible.mcp.tools import callback_list_available
        result = callback_list_available({})
        assert "callbacks" in result
        names = [d["name"] for d in result["callbacks"]]
        assert "grad_clip" in names
        assert "nan_detector" in names
        assert "early_stopping" in names


# ===================================================================
# Schema tools — verify real schemas returned
# ===================================================================

class TestSchemaTools:
    def test_optimizer_schema_valid(self):
        from crucible.mcp.tools import optimizer_get_config_schema
        result = optimizer_get_config_schema({"name": "adamw"})
        assert result["name"] == "adamw"
        assert "lr" in result["schema"]
        assert "weight_decay" in result["schema"]

    def test_optimizer_schema_unknown(self):
        from crucible.mcp.tools import optimizer_get_config_schema
        result = optimizer_get_config_schema({"name": "nonexistent_xyz"})
        assert "note" in result["schema"]  # stub response

    def test_scheduler_schema_valid(self):
        from crucible.mcp.tools import scheduler_get_config_schema
        result = scheduler_get_config_schema({"name": "cosine"})
        assert result["name"] == "cosine"
        assert "total_steps" in result["schema"]

    def test_scheduler_schema_unknown(self):
        from crucible.mcp.tools import scheduler_get_config_schema
        result = scheduler_get_config_schema({"name": "nonexistent_xyz"})
        assert "note" in result["schema"]


# ===================================================================
# Add tools — verify file creation and registration
# ===================================================================

class TestAddTools:
    def test_optimizer_add_success(self, mcp_config):
        from crucible.mcp.tools import optimizer_add
        from crucible.training.optimizers import OPTIMIZER_REGISTRY

        code = textwrap.dedent("""\
            from crucible.training.optimizers import register_optimizer
            register_optimizer("mcp_test_opt", lambda p, **kw: {"test": True}, source="local")
        """)
        try:
            result = optimizer_add({"name": "mcp_test_opt", "code": code})
            assert result["status"] == "registered"
            assert "mcp_test_opt" in OPTIMIZER_REGISTRY.list_plugins()
            # Verify file was written
            path = Path(result["path"])
            assert path.exists()
            assert "register_optimizer" in path.read_text()
        finally:
            OPTIMIZER_REGISTRY.unregister("mcp_test_opt")
            sys.modules.pop("_crucible_plugin_optimizers_local_mcp_test_opt", None)

    def test_scheduler_add_success(self, mcp_config):
        from crucible.mcp.tools import scheduler_add
        from crucible.training.schedulers import SCHEDULER_REGISTRY

        code = textwrap.dedent("""\
            from crucible.training.schedulers import register_scheduler
            register_scheduler("mcp_test_sched", lambda opt, **kw: None, source="local")
        """)
        try:
            result = scheduler_add({"name": "mcp_test_sched", "code": code})
            assert result["status"] == "registered"
            assert "mcp_test_sched" in SCHEDULER_REGISTRY.list_plugins()
        finally:
            SCHEDULER_REGISTRY.unregister("mcp_test_sched")
            sys.modules.pop("_crucible_plugin_schedulers_local_mcp_test_sched", None)

    def test_callback_add_success(self, mcp_config):
        from crucible.mcp.tools import callback_add
        from crucible.training.callbacks import CALLBACK_REGISTRY

        code = textwrap.dedent("""\
            from crucible.training.callbacks import TrainingCallback, register_callback
            class TestCB(TrainingCallback):
                pass
            register_callback("mcp_test_cb", TestCB, source="local")
        """)
        try:
            result = callback_add({"name": "mcp_test_cb", "code": code})
            assert result["status"] == "registered"
            assert "mcp_test_cb" in CALLBACK_REGISTRY.list_plugins()
        finally:
            CALLBACK_REGISTRY.unregister("mcp_test_cb")
            sys.modules.pop("_crucible_plugin_callbacks_local_mcp_test_cb", None)

    def test_logger_add_success(self, mcp_config):
        from crucible.mcp.tools import logger_add
        from crucible.runner.loggers import LOGGER_REGISTRY

        code = textwrap.dedent("""\
            from crucible.runner.loggers import TrainingLogger, register_logger
            class NullLogger(TrainingLogger):
                def log(self, metrics, *, step=None): pass
                def finish(self, exit_code=0): pass
            register_logger("mcp_test_null", NullLogger, source="local")
        """)
        try:
            result = logger_add({"name": "mcp_test_null", "code": code})
            assert result["status"] == "registered"
            assert "mcp_test_null" in LOGGER_REGISTRY.list_plugins()
        finally:
            LOGGER_REGISTRY.unregister("mcp_test_null")
            sys.modules.pop("_crucible_plugin_loggers_local_mcp_test_null", None)

    def test_provider_add_success(self, mcp_config):
        from crucible.mcp.tools import provider_add
        from crucible.fleet.provider_registry import PROVIDER_REGISTRY

        code = textwrap.dedent("""\
            from crucible.fleet.provider_registry import register_provider
            register_provider("mcp_test_prov", lambda **kw: {"mock": True}, source="local")
        """)
        try:
            result = provider_add({"name": "mcp_test_prov", "code": code})
            assert result["status"] == "registered"
            assert "mcp_test_prov" in PROVIDER_REGISTRY.list_plugins()
        finally:
            PROVIDER_REGISTRY.unregister("mcp_test_prov")
            sys.modules.pop("_crucible_plugin_providers_local_mcp_test_prov", None)

    def test_add_missing_name_returns_error(self, mcp_config):
        from crucible.mcp.tools import optimizer_add
        result = optimizer_add({"name": "", "code": "x = 1"})
        assert "error" in result

    def test_add_missing_code_returns_error(self, mcp_config):
        from crucible.mcp.tools import optimizer_add
        result = optimizer_add({"name": "foo", "code": ""})
        assert "error" in result

    def test_add_broken_code_returns_error(self, mcp_config):
        from crucible.mcp.tools import optimizer_add
        result = optimizer_add({"name": "broken", "code": "raise RuntimeError('boom')"})
        assert "error" in result

    def test_composer_add_block_type(self, mcp_config):
        from crucible.mcp.tools import composer_add_block_type
        code = "x = 'block_type_registered'\n"
        result = composer_add_block_type({"name": "test_block", "code": code})
        assert result["status"] == "registered"
        assert Path(result["path"]).exists()
        sys.modules.pop("_crucible_plugin_block_types_local_test_block", None)

    def test_composer_add_stack_pattern(self, mcp_config):
        from crucible.mcp.tools import composer_add_stack_pattern
        code = "x = 'stack_pattern_registered'\n"
        result = composer_add_stack_pattern({"name": "test_stack", "code": code})
        assert result["status"] == "registered"
        sys.modules.pop("_crucible_plugin_stack_patterns_local_test_stack", None)

    def test_composer_add_augmentation(self, mcp_config):
        from crucible.mcp.tools import composer_add_augmentation
        code = "x = 'augmentation_registered'\n"
        result = composer_add_augmentation({"name": "test_aug", "code": code})
        assert result["status"] == "registered"
        sys.modules.pop("_crucible_plugin_augmentations_local_test_aug", None)


# ===================================================================
# TOOL_DISPATCH — verify all 15 tools are wired
# ===================================================================

class TestToolDispatchWiring:
    EXPECTED_TOOLS = [
        "optimizer_list_available",
        "optimizer_add",
        "optimizer_get_config_schema",
        "scheduler_list_available",
        "scheduler_add",
        "scheduler_get_config_schema",
        "provider_list_available",
        "provider_add",
        "logger_list_available",
        "logger_add",
        "callback_list_available",
        "callback_add",
        "composer_add_block_type",
        "composer_add_stack_pattern",
        "composer_add_augmentation",
    ]

    def test_all_plugin_tools_in_dispatch(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        for tool_name in self.EXPECTED_TOOLS:
            assert tool_name in TOOL_DISPATCH, f"{tool_name} missing from TOOL_DISPATCH"
            assert callable(TOOL_DISPATCH[tool_name])

    def test_dispatch_count_is_112(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        assert len(TOOL_DISPATCH) == 112
