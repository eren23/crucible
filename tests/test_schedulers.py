"""Tests for the scheduler registry."""
from __future__ import annotations

import pytest

from crucible.core.errors import PluginError
from crucible.training.schedulers import (
    SCHEDULER_REGISTRY,
    build_scheduler,
    list_schedulers,
    list_schedulers_detailed,
    register_scheduler,
)


class TestSchedulerRegistry:
    def test_builtins_registered(self):
        names = list_schedulers()
        assert "cosine" in names
        assert "constant" in names
        assert "linear" in names
        assert "cosine_restarts" in names

    def test_list_detailed_has_source(self):
        detailed = list_schedulers_detailed()
        sources = {d["name"]: d["source"] for d in detailed}
        assert sources["cosine"] == "builtin"

    def test_build_unknown_raises(self):
        with pytest.raises(PluginError, match="Unknown scheduler"):
            # Dummy optimizer object for the call
            build_scheduler("nonexistent_sched_xyz", object())

    def test_constant_no_warmup_returns_none(self):
        result = build_scheduler("constant", object(), total_steps=100, warmup_steps=0)
        assert result is None

    def test_register_custom_scheduler(self):
        def my_sched(optimizer, **kwargs):
            return {"type": "custom_sched", "kwargs": kwargs}

        try:
            register_scheduler("test_sched", my_sched, source="local")
            assert "test_sched" in list_schedulers()
            result = build_scheduler("test_sched", object(), total_steps=100)
            assert result["type"] == "custom_sched"
        finally:
            SCHEDULER_REGISTRY.unregister("test_sched")
