"""Multi-project isolation for the fleet layer.

Two Crucible projects on the same machine share a single RUNPOD_API_KEY.
Without project-tagging, project A's ``cleanup_orphans`` would treat
project B's running pods as orphans and could destroy them. These tests
exercise the project-namespace fix:

  * Pod names are prefixed with ``{project}__``
  * ``CRUCIBLE_PROJECT`` env is injected into pod env
  * ``RunPodProvider.list_project_pods`` partitions by tag
  * ``FleetManager.cleanup_orphans`` returns sibling-project pods in
    ``legacy_pods``, never in ``orphans``, and never destroys them by
    default
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from crucible.core.config import ProjectConfig, load_config


@pytest.fixture
def fleet_config(project_dir: Path) -> ProjectConfig:
    return load_config(project_dir / "crucible.yaml")


# ---------------------------------------------------------------------------
# Pod name & env construction
# ---------------------------------------------------------------------------


class TestPodNaming:
    def test_build_pod_name_prefixes_with_project(self):
        from crucible.fleet.providers.runpod import build_pod_name

        assert build_pod_name("alpha", "wave1", 3) == "alpha__wave1-03"
        assert build_pod_name("beta-2", "screen", 12) == "beta-2__screen-12"

    def test_build_pod_name_falls_back_when_project_empty(self):
        from crucible.fleet.providers.runpod import build_pod_name

        assert build_pod_name("", "wave1", 3) == "wave1-03"
        assert build_pod_name(None, "wave1", 3) == "wave1-03"  # type: ignore[arg-type]

    def test_normalize_project_name_strips_unsafe_chars(self):
        from crucible.fleet.providers.runpod import normalize_project_name

        assert normalize_project_name("parameter-golf_dev") == "parameter-golf_dev"
        assert normalize_project_name("foo bar/baz") == "foo-bar-baz"
        assert normalize_project_name("") == ""

    def test_is_project_pod(self):
        from crucible.fleet.providers.runpod import is_project_pod

        assert is_project_pod("alpha__wave1-03", "alpha")
        assert not is_project_pod("beta__wave1-03", "alpha")
        assert not is_project_pod("legacy-pod-01", "alpha")
        # Empty project never matches anything (legacy pods get untagged)
        assert not is_project_pod("anything", "")

    def test_no_prefix_collision_between_projects_with_shared_root(self):
        """Project names sharing a prefix must NOT match each other's pods.

        If we used a single ``-`` as separator, ``crucible-`` would prefix-
        match ``crucible-extra-wave1-01`` belonging to a sibling. The double-
        underscore separator prevents that.
        """
        from crucible.fleet.providers.runpod import is_project_pod

        a_name = "crucible__wave1-01"
        b_name = "crucible-extra__wave1-01"
        assert is_project_pod(a_name, "crucible")
        assert not is_project_pod(b_name, "crucible")
        assert is_project_pod(b_name, "crucible-extra")

    def test_is_project_pod_exact_match_not_prefix(self):
        """`foo` must NOT claim pods belonging to `foo__bar` (which after
        collapse normalizes to `foo_bar` and produces pods like
        `foo_bar__wave-01`). Pod names with TWO separators (e.g. a
        manually-crafted `foo__bar__wave-01`) are refused outright as
        defense in depth — they cannot be produced by any legitimate
        project."""
        from crucible.fleet.providers.runpod import is_project_pod

        # Manual / two-separator pod: never claimed by anyone.
        manual_pod = "foo__bar__wave-01"
        assert not is_project_pod(manual_pod, "foo")
        assert not is_project_pod(manual_pod, "foo__bar")
        # Real `foo__bar` project produces pods named `foo_bar__...`
        # (separator collapse), and that DOES match.
        assert is_project_pod("foo_bar__wave-01", "foo__bar")
        assert is_project_pod("foo_bar__wave-01", "foo_bar")
        # And `foo` matches its own `foo__...` pods.
        assert is_project_pod("foo__wave-01", "foo")
        # But not `foo_bar`'s pods.
        assert not is_project_pod("foo_bar__wave-01", "foo")

    def test_normalize_collapses_double_underscores(self):
        """Project names with `__` must collapse so the separator is unambiguous."""
        from crucible.fleet.providers.runpod import normalize_project_name

        assert normalize_project_name("foo__bar") == "foo_bar"
        assert normalize_project_name("a___b____c") == "a_b_c"

    def test_name_prefix_with_separator_is_rejected(self):
        """A `name_prefix` containing `__` would let sibling projects
        claim our pods; reject early."""
        from crucible.core.errors import FleetError
        from crucible.fleet.providers.runpod import build_pod_name

        with pytest.raises(FleetError, match="separator"):
            build_pod_name("alpha", "wave__retry", 1)

    def test_next_node_index_recognizes_tagged_pods(self):
        """`next_node_index` must strip the `{project}__` tag before
        matching `name_prefix`, otherwise replacement provisioning
        always returns 1 and produces duplicate ordinals."""
        from crucible.fleet.inventory import next_node_index

        nodes = [
            {"name": "alpha__crucible-day-01"},
            {"name": "alpha__crucible-day-02"},
            {"name": "alpha__crucible-day-03"},
        ]
        assert next_node_index(nodes, "crucible-day") == 4
        # Mixed tagged + legacy
        nodes_mixed = [
            {"name": "crucible-day-01"},
            {"name": "alpha__crucible-day-02"},
        ]
        assert next_node_index(nodes_mixed, "crucible-day") == 3


class TestPodEnvInjection:
    def test_create_api_pod_payload_carries_crucible_project_env(self, monkeypatch):
        """``CRUCIBLE_PROJECT`` env is added when project_name is set."""
        from crucible.fleet.providers import runpod as rp_module

        captured: dict[str, Any] = {}

        def fake_graphql(query, variables=None, timeout=None):
            captured["input"] = variables["input"] if variables else {}
            return {
                "podFindAndDeployInterruptable": {
                    "id": "pod-1",
                    "name": variables["input"]["name"],
                    "desiredStatus": "RUNNING",
                    "imageName": variables["input"]["imageName"],
                    "machine": {"podHostId": "host-1"},
                },
            }

        monkeypatch.setattr(rp_module, "runpod_graphql", fake_graphql)
        rp_module.create_api_pod(
            name="alpha__wave1-01",
            gpu_type_ids=["NVIDIA GeForce RTX 4090"],
            image_name="img:latest",
            cloud_type="COMMUNITY",
            interruptible=True,
            container_disk_gb=20,
            volume_gb=40,
            volume_mount_path="/workspace",
            public_key="ssh-ed25519 AAAA...",
            ports=["22/tcp"],
            project_name="alpha",
        )
        env = captured["input"]["env"]
        env_keys = {item["key"] for item in env}
        assert "CRUCIBLE_PROJECT" in env_keys
        env_map = {item["key"]: item["value"] for item in env}
        assert env_map["CRUCIBLE_PROJECT"] == "alpha"

    def test_create_api_pod_omits_project_env_when_unset(self, monkeypatch):
        from crucible.fleet.providers import runpod as rp_module

        captured: dict[str, Any] = {}

        def fake_graphql(query, variables=None, timeout=None):
            captured["input"] = variables["input"] if variables else {}
            return {
                "podFindAndDeployInterruptable": {
                    "id": "pod-1",
                    "name": variables["input"]["name"],
                    "desiredStatus": "RUNNING",
                    "imageName": variables["input"]["imageName"],
                    "machine": {"podHostId": "host-1"},
                },
            }

        monkeypatch.setattr(rp_module, "runpod_graphql", fake_graphql)
        rp_module.create_api_pod(
            name="legacy-01",
            gpu_type_ids=["NVIDIA GeForce RTX 4090"],
            image_name="img:latest",
            cloud_type="COMMUNITY",
            interruptible=True,
            container_disk_gb=20,
            volume_gb=40,
            volume_mount_path="/workspace",
            public_key="ssh-ed25519 AAAA...",
            ports=["22/tcp"],
        )
        env_keys = {item["key"] for item in captured["input"]["env"]}
        assert "CRUCIBLE_PROJECT" not in env_keys


# ---------------------------------------------------------------------------
# RunPodProvider.list_project_pods
# ---------------------------------------------------------------------------


class TestListProjectPods:
    def test_partitions_by_name_prefix(self, monkeypatch):
        from crucible.fleet.providers.runpod import RunPodProvider

        provider = RunPodProvider(project_name="alpha")
        monkeypatch.setattr(
            provider, "list_all_pods",
            lambda: [
                {"id": "1", "name": "alpha__wave1-01"},
                {"id": "2", "name": "alpha__wave1-02"},
                {"id": "3", "name": "beta__wave1-01"},
                {"id": "4", "name": "legacy-pod"},
            ],
        )
        partition = provider.list_project_pods()
        assert {p["id"] for p in partition["tagged"]} == {"1", "2"}
        assert {p["id"] for p in partition["untagged"]} == {"3", "4"}

    def test_explicit_project_overrides_constructor(self, monkeypatch):
        from crucible.fleet.providers.runpod import RunPodProvider

        provider = RunPodProvider(project_name="alpha")
        monkeypatch.setattr(
            provider, "list_all_pods",
            lambda: [
                {"id": "1", "name": "alpha__wave1-01"},
                {"id": "2", "name": "beta__wave1-01"},
            ],
        )
        partition = provider.list_project_pods(project_name="beta")
        assert {p["id"] for p in partition["tagged"]} == {"2"}
        assert {p["id"] for p in partition["untagged"]} == {"1"}

    def test_no_project_returns_everything_as_tagged(self, monkeypatch):
        """With no project namespace, fall back to legacy semantics."""
        from crucible.fleet.providers.runpod import RunPodProvider

        provider = RunPodProvider(project_name="")
        pods = [
            {"id": "1", "name": "wave1-01"},
            {"id": "2", "name": "wave1-02"},
        ]
        monkeypatch.setattr(provider, "list_all_pods", lambda: pods)
        partition = provider.list_project_pods()
        assert partition["tagged"] == pods
        assert partition["untagged"] == []


# ---------------------------------------------------------------------------
# FleetManager.cleanup_orphans cross-project safety
# ---------------------------------------------------------------------------


class TestCleanupOrphansCrossProject:
    """conftest sets project name to ``test-project``."""

    def test_sibling_project_pods_appear_as_legacy_not_orphans(
        self, fleet_config: ProjectConfig,
    ):
        """The headline failure mode: project A must NOT see project B's pods
        as orphans, and ``destroy=True`` must NOT destroy them."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        fm._provider = MagicMock()
        # Account holds: 2 our-project orphans, 3 sibling-project pods, 1 untagged
        fm._provider.list_project_pods.return_value = {
            "tagged": [
                {"id": "ours-1", "name": "test-project__wave1-01"},
                {"id": "ours-2", "name": "test-project__wave1-02"},
            ],
            "untagged": [
                {"id": "sib-1", "name": "sfumato__wave1-01"},
                {"id": "sib-2", "name": "sfumato__wave1-02"},
                {"id": "sib-3", "name": "sfumato__wave1-03"},
                {"id": "leg-1", "name": "manual-pod"},
            ],
        }
        fm._provider.destroy_pods_by_id.return_value = ["ours-1", "ours-2"]

        result = fm.cleanup_orphans(destroy=True)

        assert {o["pod_id"] for o in result["orphans"]} == {"ours-1", "ours-2"}
        assert {o["pod_id"] for o in result["legacy_pods"]} == {
            "sib-1", "sib-2", "sib-3", "leg-1",
        }
        # CRITICAL invariant: destroy must only touch tagged orphans
        called_ids = set(fm._provider.destroy_pods_by_id.call_args[0][0])
        assert called_ids == {"ours-1", "ours-2"}
        assert "sib-1" not in called_ids
        assert "leg-1" not in called_ids

    def test_include_legacy_opts_in_to_destroying_siblings(
        self, fleet_config: ProjectConfig,
    ):
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        fm._provider = MagicMock()
        fm._provider.list_project_pods.return_value = {
            "tagged": [{"id": "ours-1", "name": "test-project__wave1-01"}],
            "untagged": [{"id": "sib-1", "name": "sfumato__wave1-01"}],
        }
        fm._provider.destroy_pods_by_id.return_value = ["ours-1", "sib-1"]

        result = fm.cleanup_orphans(destroy=True, include_legacy=True)

        called_ids = set(fm._provider.destroy_pods_by_id.call_args[0][0])
        assert called_ids == {"ours-1", "sib-1"}
        assert set(result["destroyed"]) == {"ours-1", "sib-1"}

    def test_legacy_pods_still_listed_when_destroy_false(
        self, fleet_config: ProjectConfig,
    ):
        """destroy=False is purely informational: legacy pods are surfaced
        for visibility but never touched."""
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        fm._provider = MagicMock()
        fm._provider.list_project_pods.return_value = {
            "tagged": [],
            "untagged": [{"id": "sib-1", "name": "sfumato__wave1-01"}],
        }

        result = fm.cleanup_orphans(destroy=False)

        assert result["orphans"] == []
        assert len(result["legacy_pods"]) == 1
        assert result["legacy_pods"][0]["pod_id"] == "sib-1"
        fm._provider.destroy_pods_by_id.assert_not_called()


# ---------------------------------------------------------------------------
# Provider factory plumbing
# ---------------------------------------------------------------------------


class TestRefreshDoesNotAdoptSiblingPods:
    """Reviewer-flagged regression: ``refresh()`` previously called
    ``list_all_pods()`` for orphan reconciliation and adopted any account
    pod into ``nodes.json`` as ``state="reconciled_orphan"``. With the
    fix, only this project's tagged pods get reconciled."""

    def test_refresh_only_reconciles_tagged_pods(self, monkeypatch):
        from crucible.fleet.providers.runpod import RunPodProvider

        provider = RunPodProvider(project_name="alpha")
        # Bypass real RunPod API
        monkeypatch.setattr(
            provider, "list_all_pods",
            lambda: [
                {"id": "ours-1", "name": "alpha__wave-01"},
                {"id": "sib-1", "name": "beta__wave-01"},
                {"id": "leg-1", "name": "manual-pod"},
            ],
        )
        # No previously-known nodes → all three pods are "unseen" by refresh
        refreshed = provider.refresh([])
        names = {n.get("name") for n in refreshed if n.get("state") == "reconciled_orphan"}
        # Only the tagged pod is adopted.
        assert "alpha__wave-01" in names
        assert "beta__wave-01" not in names
        assert "manual-pod" not in names

    def test_refresh_with_no_project_falls_back_to_legacy_reconcile(self, monkeypatch):
        """Single-project users (no project_name set) keep the old
        all-pods reconciliation behavior."""
        from crucible.fleet.providers.runpod import RunPodProvider

        provider = RunPodProvider(project_name="")
        monkeypatch.setattr(
            provider, "list_all_pods",
            lambda: [
                {"id": "p1", "name": "wave-01"},
                {"id": "p2", "name": "wave-02"},
            ],
        )
        refreshed = provider.refresh([])
        names = {n.get("name") for n in refreshed if n.get("state") == "reconciled_orphan"}
        assert names == {"wave-01", "wave-02"}


class TestDestroyNodesProjectScoped:
    """Reviewer-flagged blocker: ``destroy_nodes`` with no args used to
    call ``destroy_all_pods()`` — wiping every pod on the RunPod account
    including sibling projects'. Now it cascades through cleanup_orphans,
    which is project-scoped."""

    def test_no_arg_destroy_only_touches_tagged_pods(self, monkeypatch, fleet_config):
        from crucible.fleet.manager import FleetManager
        from crucible.fleet.providers.runpod import RunPodProvider
        from crucible.mcp import tools

        # Force runpod path
        fleet_config.provider.type = "runpod"
        fm = FleetManager(fleet_config)
        fm.nodes_file.write_text("[]")

        provider = MagicMock(spec=RunPodProvider)
        provider.list_project_pods.return_value = {
            "tagged": [{"id": "ours-1", "name": "test-project__wave-01"}],
            "untagged": [{"id": "sib-1", "name": "sfumato__wave-01"}],
        }
        provider.destroy_pods_by_id.return_value = ["ours-1"]
        fm._provider = provider

        # Patch the MCP module's helpers to use our manager
        monkeypatch.setattr(tools, "_get_config", lambda: fleet_config)
        monkeypatch.setattr(tools, "_get_fleet_manager", lambda cfg: fm)

        # Mock fleet.destroy (inventory destroy) so we don't try real SSH
        fm.destroy = MagicMock(return_value=[])

        result = tools.destroy_nodes({})

        # Only our project's pod was destroyed
        called_ids = set(provider.destroy_pods_by_id.call_args[0][0])
        assert called_ids == {"ours-1"}
        assert "sib-1" not in called_ids
        # Provider.destroy_all_pods is NEVER called from this path
        assert not provider.destroy_all_pods.called
        assert "ok" in result.get("status", "")


class TestProviderFactoryPlumbing:
    def test_runpod_factory_passes_project_name(self):
        from crucible.fleet.provider_registry import build_provider

        provider = build_provider(
            "runpod",
            ssh_key="~/.ssh/id_ed25519_runpod",
            project_name="my-project",
        )
        assert provider.project_name == "my-project"

    def test_fleet_manager_threads_project_name_to_provider(
        self, fleet_config: ProjectConfig, monkeypatch,
    ):
        """``FleetManager`` constructed from ``ProjectConfig.name`` should
        produce a provider tagged with that name."""
        from crucible.fleet import manager as mgr_module
        from crucible.fleet.manager import FleetManager

        # conftest's sample yaml uses provider.type=ssh, swap to runpod via
        # monkeypatch to exercise the runpod factory.
        fleet_config.provider.type = "runpod"
        fm = FleetManager(fleet_config)
        provider = fm.provider
        assert getattr(provider, "project_name", None) == "test-project"
