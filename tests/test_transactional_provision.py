"""Tests for transactional provisioning + orphan cleanup (Phase 2a).

These tests exercise the root fix for the provision_project orphan bug:
when a multi-pod provision batch fails partway through, the successfully-
created pods are committed to inventory before the error is re-raised, so
they are never orphaned on the provider side.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from crucible.core.config import ProjectConfig, load_config
from crucible.core.errors import PartialProvisionError


@pytest.fixture
def fleet_config(project_dir: Path) -> ProjectConfig:
    return load_config(project_dir / "crucible.yaml")


def _make_fake_node(name: str, pod_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "node_id": pod_id,
        "pod_id": pod_id,
        "state": "creating",
        "ssh_host": "",
        "ssh_port": 22,
    }


# ---------------------------------------------------------------------------
# PartialProvisionError shape
# ---------------------------------------------------------------------------


class TestPartialProvisionError:
    def test_carries_created_and_failed(self):
        created = [{"name": "a-1", "pod_id": "p1"}]
        failed = [{"name": "a-2", "error": "api timeout"}]
        exc = PartialProvisionError("oops", created=created, failed=failed)
        assert exc.created == created
        assert exc.failed == failed
        assert "oops" in str(exc)

    def test_defaults_to_empty_lists(self):
        exc = PartialProvisionError("msg")
        assert exc.created == []
        assert exc.failed == []

    def test_is_a_fleet_error(self):
        from crucible.core.errors import FleetError
        exc = PartialProvisionError("msg")
        assert isinstance(exc, FleetError)


# ---------------------------------------------------------------------------
# FleetManager.provision persistence on partial failure
# ---------------------------------------------------------------------------


class TestProvisionPartialFailure:
    def test_commits_created_nodes_before_reraising(
        self, fleet_config: ProjectConfig
    ):
        """The canonical provision_project orphan scenario."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        created = [
            _make_fake_node("batch-01", "p1"),
            _make_fake_node("batch-02", "p2"),
        ]
        failed = [{"name": "batch-03", "error": "no capacity"}]

        fm._provider = MagicMock()
        fm._provider.provision.side_effect = PartialProvisionError(
            "2/3 created", created=created, failed=failed
        )

        with pytest.raises(PartialProvisionError) as exc_info:
            fm.provision(count=3, name_prefix="batch")

        # The 2 successful pods MUST be in inventory despite the exception
        saved = json.loads(fm.nodes_file.read_text())
        saved_names = {n["name"] for n in saved}
        assert "batch-01" in saved_names
        assert "batch-02" in saved_names
        assert "batch-03" not in saved_names

        # And the exception still carries the partial state for the caller
        assert len(exc_info.value.created) == 2
        assert len(exc_info.value.failed) == 1

    def test_merges_partial_with_existing(self, fleet_config: ProjectConfig):
        """Partial-success commit should not destroy pre-existing nodes."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        existing = [{"name": "old-1", "node_id": "old", "pod_id": "old", "state": "ready"}]
        fm.nodes_file.write_text(json.dumps(existing))

        fm._provider = MagicMock()
        fm._provider.provision.side_effect = PartialProvisionError(
            "1/2 created",
            created=[_make_fake_node("new-1", "new")],
            failed=[{"name": "new-2", "error": "quota"}],
        )

        with pytest.raises(PartialProvisionError):
            fm.provision(count=2, name_prefix="new")

        saved = json.loads(fm.nodes_file.read_text())
        names = {n["name"] for n in saved}
        assert names == {"old-1", "new-1"}

    def test_empty_created_still_reraises(self, fleet_config: ProjectConfig):
        """If every pod failed, nothing is committed but the error still propagates."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        fm._provider = MagicMock()
        fm._provider.provision.side_effect = PartialProvisionError(
            "0/2 created",
            created=[],
            failed=[
                {"name": "z-01", "error": "quota"},
                {"name": "z-02", "error": "quota"},
            ],
        )

        with pytest.raises(PartialProvisionError):
            fm.provision(count=2, name_prefix="z")

        saved = json.loads(fm.nodes_file.read_text())
        assert saved == []


# ---------------------------------------------------------------------------
# FleetManager.cleanup_orphans
# ---------------------------------------------------------------------------


class TestCleanupOrphans:
    """conftest project name is ``test-project``; project-tagged pods carry
    the prefix ``test-project__``."""

    def test_lists_orphans_without_destroying(self, fleet_config: ProjectConfig):
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text(
            json.dumps([{"name": "test-project__tracked", "node_id": "tracked-id", "pod_id": "tracked-id"}])
        )

        # Provider reports two project-tagged pods — one tracked, one orphan
        fm._provider = MagicMock()
        fm._provider.list_project_pods.return_value = {
            "tagged": [
                {"id": "tracked-id", "name": "test-project__tracked"},
                {"id": "orphan-id", "name": "test-project__runaway"},
            ],
            "untagged": [],
        }

        result = fm.cleanup_orphans(destroy=False)

        assert len(result["orphans"]) == 1
        assert result["orphans"][0]["pod_id"] == "orphan-id"
        assert result["orphans"][0]["name"] == "test-project__runaway"
        assert result["destroyed"] == []
        assert result["legacy_pods"] == []
        fm._provider.destroy_pods_by_id.assert_not_called()

    def test_destroys_orphans_when_requested(self, fleet_config: ProjectConfig):
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        fm._provider = MagicMock()
        fm._provider.list_project_pods.return_value = {
            "tagged": [
                {"id": "orphan-1", "name": "test-project__a"},
                {"id": "orphan-2", "name": "test-project__b"},
            ],
            "untagged": [],
        }
        fm._provider.destroy_pods_by_id.return_value = ["orphan-1", "orphan-2"]

        result = fm.cleanup_orphans(destroy=True)

        fm._provider.destroy_pods_by_id.assert_called_once()
        call_args = fm._provider.destroy_pods_by_id.call_args[0][0]
        assert set(call_args) == {"orphan-1", "orphan-2"}
        assert set(result["destroyed"]) == {"orphan-1", "orphan-2"}

    def test_no_orphans_returns_empty(self, fleet_config: ProjectConfig):
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text(
            json.dumps([{"name": "test-project__t", "node_id": "a", "pod_id": "a"}])
        )

        fm._provider = MagicMock()
        fm._provider.list_project_pods.return_value = {
            "tagged": [{"id": "a", "name": "test-project__t"}],
            "untagged": [],
        }

        result = fm.cleanup_orphans(destroy=True)
        assert result["orphans"] == []
        assert result["destroyed"] == []
        fm._provider.destroy_pods_by_id.assert_not_called()

    def test_raises_when_provider_does_not_support_listing(
        self, fleet_config: ProjectConfig
    ):
        from crucible.core.errors import FleetError
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        # MagicMock has list_all_pods callable by default — delete it to
        # simulate a provider (e.g. SSH) that doesn't expose it.
        fm._provider = MagicMock(spec=[])

        with pytest.raises(FleetError, match="does not support"):
            fm.cleanup_orphans()

    def test_uses_node_id_fallback_for_tracking(self, fleet_config: ProjectConfig):
        """Nodes with only node_id (no pod_id) should still count as tracked."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text(
            json.dumps([{"name": "test-project__old-style", "node_id": "legacy-id"}])
        )

        fm._provider = MagicMock()
        fm._provider.list_project_pods.return_value = {
            "tagged": [{"id": "legacy-id", "name": "test-project__old-style"}],
            "untagged": [],
        }

        result = fm.cleanup_orphans(destroy=False)
        assert result["orphans"] == []


# ---------------------------------------------------------------------------
# RunPodProvider.provision raises PartialProvisionError with mixed state
# ---------------------------------------------------------------------------


class TestRunPodProvisionBatching:
    def test_mixed_batch_raises_partial_error(self, monkeypatch):
        """With the runpod API mocked, a 2-success-1-fail batch must raise
        PartialProvisionError carrying 2 created + 1 failed.
        """
        from crucible.core.errors import FleetError
        from crucible.fleet.providers import runpod as rp_module

        call_count = {"n": 0}

        def fake_create_api_pod(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise FleetError("no capacity")
            return {
                "id": f"fake-{call_count['n']}",
                "name": kwargs["name"],
                "machine": {"gpuDisplayName": "RTX 4090"},
                "desiredStatus": "RUNNING",
            }

        monkeypatch.setattr(rp_module, "create_api_pod", fake_create_api_pod)
        monkeypatch.setattr(
            rp_module,
            "read_public_key",
            lambda path: "ssh-rsa FAKE KEY",
        )

        provider = rp_module.RunPodProvider(
            ssh_key="~/.ssh/fake",
            public_key_path="~/.ssh/fake.pub",
            gpu_type_ids=["NVIDIA GeForce RTX 4090"],
            image_name="test:image",
            container_disk_gb=20,
            volume_gb=0,
            volume_mount_path="/workspace",
            ports=["22/tcp"],
            interruptible=False,
            cloud_types=["SECURE"],
            gpu_count=1,
            defaults={},
            network_volume_id="",
            template_id="",
        )

        with pytest.raises(PartialProvisionError) as exc_info:
            provider.provision(count=3, name_prefix="test")

        # 2 successes, 1 failure
        assert len(exc_info.value.created) == 2
        assert len(exc_info.value.failed) == 1
        # Failed pod name should reflect the batch index (03 was the third)
        assert exc_info.value.failed[0]["name"] == "test-03"
        # Successful pods should have their RunPod state
        for node in exc_info.value.created:
            assert node["state"] == "creating"

    def test_all_succeed_returns_normally(self, monkeypatch):
        """When every pod creates cleanly, no error is raised."""
        from crucible.fleet.providers import runpod as rp_module

        counter = {"n": 0}

        def fake_create(**kwargs):
            counter["n"] += 1
            return {
                "id": f"ok-{counter['n']}",
                "name": kwargs["name"],
                "machine": {"gpuDisplayName": "RTX 4090"},
                "desiredStatus": "RUNNING",
            }

        monkeypatch.setattr(rp_module, "create_api_pod", fake_create)
        monkeypatch.setattr(
            rp_module, "read_public_key", lambda path: "ssh-rsa FAKE KEY"
        )

        provider = rp_module.RunPodProvider(
            ssh_key="~/.ssh/fake",
            public_key_path="~/.ssh/fake.pub",
            gpu_type_ids=["NVIDIA GeForce RTX 4090"],
            image_name="test:image",
            container_disk_gb=20,
            volume_gb=0,
            volume_mount_path="/workspace",
            ports=["22/tcp"],
            interruptible=False,
            cloud_types=["SECURE"],
            gpu_count=1,
            defaults={},
            network_volume_id="",
            template_id="",
        )

        result = provider.provision(count=2, name_prefix="clean")
        assert len(result) == 2
        assert all(n["state"] == "creating" for n in result)
