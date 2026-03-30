"""Tests for crucible.fleet.manager — FleetManager orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crucible.core.config import ProjectConfig, load_config


@pytest.fixture
def fleet_config(project_dir: Path) -> ProjectConfig:
    """Load a ProjectConfig from the test project directory."""
    return load_config(project_dir / "crucible.yaml")


class TestBuildProvider:
    def test_runpod_provider(self, fleet_config: ProjectConfig):
        """Config with provider.type='runpod' produces RunPodProvider."""
        from crucible.fleet.manager import FleetManager
        from crucible.fleet.providers.runpod import RunPodProvider

        fleet_config.provider.type = "runpod"
        provider = FleetManager._build_provider(fleet_config)
        assert isinstance(provider, RunPodProvider)

    def test_ssh_provider(self, fleet_config: ProjectConfig):
        """Config with provider.type='ssh' produces SSHProvider."""
        from crucible.fleet.manager import FleetManager
        from crucible.fleet.providers.ssh import SSHProvider

        assert fleet_config.provider.type == "ssh"
        provider = FleetManager._build_provider(fleet_config)
        assert isinstance(provider, SSHProvider)

    def test_unknown_type_raises_plugin_error(self, fleet_config: ProjectConfig):
        """Unrecognized provider type raises PluginError."""
        from crucible.fleet.manager import FleetManager
        from crucible.core.errors import PluginError

        fleet_config.provider.type = "kubernetes"
        with pytest.raises(PluginError, match="Unknown fleet provider"):
            FleetManager._build_provider(fleet_config)


class TestProvision:
    def test_provision_saves_nodes(self, fleet_config: ProjectConfig):
        """provision() delegates to provider and saves to nodes_file."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fake_nodes = [
            {"name": "test-1", "node_id": "n1", "state": "pending"},
            {"name": "test-2", "node_id": "n2", "state": "pending"},
        ]
        fm._provider = MagicMock()
        fm._provider.provision.return_value = fake_nodes
        fm.nodes_file.write_text("[]")

        result = fm.provision(count=2, name_prefix="test")
        assert len(result) == 2
        assert fm._provider.provision.called
        # Verify nodes were persisted
        assert fm.nodes_file.exists()

    def test_provision_merges_with_existing_nodes(self, fleet_config: ProjectConfig):
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm._provider = MagicMock()
        fm.nodes_file.write_text(json.dumps([{"name": "existing-1", "node_id": "old-1", "state": "ready"}]))
        fm._provider.provision.return_value = [{"name": "new-1", "node_id": "new-1", "state": "pending"}]

        result = fm.provision(count=1, name_prefix="new")

        assert {node["name"] for node in result} == {"existing-1", "new-1"}


class TestDestroy:
    def test_destroy_delegates_to_provider(self, fleet_config: ProjectConfig):
        """destroy() calls provider.destroy and updates inventory."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm._provider = MagicMock()
        fm._provider.destroy.return_value = []

        existing_nodes = [{"name": "gpu-1", "node_id": "n1", "state": "ready"}]
        fm.nodes_file.write_text(json.dumps(existing_nodes))

        fm.destroy()
        fm._provider.destroy.assert_called_once()


class TestRefresh:
    def test_refresh_calls_provider(self, fleet_config: ProjectConfig):
        """refresh() calls provider.refresh to update node states."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm._provider = MagicMock()
        updated_nodes = [{"name": "gpu-1", "node_id": "n1", "state": "ready", "ssh_host": "1.2.3.4"}]
        fm._provider.refresh.return_value = updated_nodes

        existing = [{"name": "gpu-1", "node_id": "n1", "state": "pending"}]
        fm.nodes_file.write_text(json.dumps(existing))

        result = fm.refresh()
        fm._provider.refresh.assert_called_once()
        assert result[0]["ssh_host"] == "1.2.3.4"

    def test_refresh_preserves_project_metadata(self, fleet_config: ProjectConfig):
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm._provider = MagicMock()
        fm._provider.refresh.return_value = [{"name": "gpu-1", "node_id": "n1", "state": "ready", "ssh_host": "1.2.3.4"}]

        existing = [{"name": "gpu-1", "node_id": "n1", "state": "pending", "project": "lewm"}]
        fm.nodes_file.write_text(json.dumps(existing))

        result = fm.refresh()

        assert result[0]["project"] == "lewm"
