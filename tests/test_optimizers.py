"""Tests for the optimizer registry."""
from __future__ import annotations

import pytest

from crucible.core.errors import PluginError
from crucible.training.optimizers import (
    OPTIMIZER_REGISTRY,
    build_optimizer,
    list_optimizers,
    list_optimizers_detailed,
    register_optimizer,
)


class TestOptimizerRegistry:
    def test_builtins_registered(self):
        names = list_optimizers()
        assert "adam" in names
        assert "adamw" in names
        assert "sgd" in names
        assert "muon" in names
        assert "rmsprop" in names

    def test_list_detailed_has_source(self):
        detailed = list_optimizers_detailed()
        sources = {d["name"]: d["source"] for d in detailed}
        assert sources["adam"] == "builtin"
        assert sources["adamw"] == "builtin"

    def test_build_unknown_raises(self):
        with pytest.raises(PluginError, match="Unknown optimizer"):
            build_optimizer("nonexistent_optimizer_xyz", [])

    def test_register_custom_optimizer(self):
        def my_factory(params, **kwargs):
            return {"type": "custom", "params": params, "kwargs": kwargs}

        try:
            register_optimizer("test_custom", my_factory, source="local")
            assert "test_custom" in list_optimizers()
            result = build_optimizer("test_custom", [1, 2, 3], lr=0.01)
            assert result["type"] == "custom"
            assert result["kwargs"]["lr"] == 0.01
        finally:
            OPTIMIZER_REGISTRY.unregister("test_custom")
