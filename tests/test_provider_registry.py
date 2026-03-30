"""Tests for the fleet provider registry."""
from __future__ import annotations

import pytest

from crucible.core.errors import PluginError
from crucible.fleet.provider_registry import (
    PROVIDER_REGISTRY,
    build_provider,
    list_providers,
    list_providers_detailed,
    register_provider,
)


class TestProviderRegistry:
    def test_builtins_registered(self):
        names = list_providers()
        assert "runpod" in names
        assert "ssh" in names

    def test_list_detailed_has_source(self):
        detailed = list_providers_detailed()
        sources = {d["name"]: d["source"] for d in detailed}
        assert sources["runpod"] == "builtin"
        assert sources["ssh"] == "builtin"

    def test_build_unknown_raises(self):
        with pytest.raises(PluginError, match="Unknown fleet provider"):
            build_provider("nonexistent_provider")

    def test_register_custom_provider(self):
        def mock_factory(**kwargs):
            return {"type": "mock", "kwargs": kwargs}

        try:
            register_provider("mock_cloud", mock_factory, source="local")
            assert "mock_cloud" in list_providers()
            result = build_provider("mock_cloud", ssh_key="/tmp/key")
            assert result["type"] == "mock"
            assert result["kwargs"]["ssh_key"] == "/tmp/key"
        finally:
            PROVIDER_REGISTRY.unregister("mock_cloud")

    def test_runpod_build_threads_interruptible(self):
        provider = build_provider("runpod", interruptible=False)
        assert provider.interruptible is False
        assert provider.cloud_types == ["SECURE", "COMMUNITY"]
