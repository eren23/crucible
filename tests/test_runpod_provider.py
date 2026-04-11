"""Tests for RunPod provider payload validation, stop/start, and inventory record state machine."""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from crucible.core.errors import FleetError
from crucible.fleet.providers.runpod import (
    RunPodProvider,
    build_pod_payload,
    inventory_record_from_api,
)


class TestBuildPodPayload:
    def test_rejects_volume_smaller_than_container_disk(self):
        with pytest.raises(FleetError, match="volume size must be greater than or equal"):
            build_pod_payload(
                name="lewm-01",
                gpu_type_ids=["NVIDIA GeForce RTX 4090"],
                image_name="runpod/pytorch:latest",
                cloud_type="SECURE",
                interruptible=False,
                container_disk_gb=80,
                volume_gb=40,
                volume_mount_path="/workspace",
                public_key="ssh-ed25519 AAAA test",
                ports=["22/tcp"],
            )

    def test_accepts_valid_disk_sizes(self):
        payload = build_pod_payload(
            name="lewm-01",
            gpu_type_ids=["NVIDIA GeForce RTX 4090"],
            image_name="runpod/pytorch:latest",
            cloud_type="SECURE",
            interruptible=False,
            container_disk_gb=40,
            volume_gb=80,
            volume_mount_path="/workspace",
            public_key="ssh-ed25519 AAAA test",
            ports=["22/tcp"],
        )

        assert payload["containerDiskInGb"] == 40
        assert payload["volumeInGb"] == 80


# ---------------------------------------------------------------------------
# inventory_record_from_api — state machine transitions
# ---------------------------------------------------------------------------

class TestInventoryRecordStateTransitions:
    """Verify the stop/start lifecycle state machine in inventory_record_from_api."""

    def test_running_pod_new_record_gets_api_state(self):
        """No previous record, API reports 'running' -> state='running'."""
        raw = {"id": "pod_123", "desiredStatus": "RUNNING"}
        rec = inventory_record_from_api(raw)
        assert rec["state"] == "running"
        assert rec["api_state"] == "running"

    def test_stopped_api_overrides_ready_local(self):
        """previous.state='ready', API reports 'stopped' -> state='stopped'."""
        raw = {"id": "pod_123", "desiredStatus": "STOPPED"}
        previous = {"state": "ready"}
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["state"] == "stopped"

    def test_stopped_api_does_not_override_starting(self):
        """previous.state='starting', API reports 'stopped' -> state stays 'starting'."""
        raw = {"id": "pod_123", "desiredStatus": "STOPPED"}
        previous = {"state": "starting"}
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["state"] == "starting"

    def test_stopped_local_running_api_becomes_starting(self):
        """previous.state='stopped', API reports 'running' -> state='starting'."""
        raw = {"id": "pod_123", "desiredStatus": "RUNNING"}
        previous = {"state": "stopped"}
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["state"] == "starting"

    def test_starting_local_running_api_becomes_new(self):
        """previous.state='starting', API reports 'running' -> state='new'."""
        raw = {"id": "pod_123", "desiredStatus": "RUNNING"}
        previous = {"state": "starting"}
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["state"] == "new"

    def test_bootstrap_flags_preserved_through_stop(self):
        """Previous has bootstrap flags set, API reports 'stopped' -> flags preserved."""
        raw = {"id": "pod_123", "desiredStatus": "STOPPED"}
        previous = {
            "state": "ready",
            "env_ready": True,
            "dataset_ready": True,
            "git_sha": "abc",
        }
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["state"] == "stopped"
        assert rec["env_ready"] is True
        assert rec["dataset_ready"] is True
        assert rec["git_sha"] == "abc"

    def test_network_volume_id_from_api(self):
        """Raw response has networkVolumeId -> node record gets it."""
        raw = {"id": "pod_123", "networkVolumeId": "vol_123"}
        rec = inventory_record_from_api(raw)
        assert rec["network_volume_id"] == "vol_123"

    def test_network_volume_id_from_previous(self):
        """Raw has no networkVolumeId, previous has it -> preserved."""
        raw = {"id": "pod_123"}
        previous = {"network_volume_id": "vol_456"}
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["network_volume_id"] == "vol_456"


class TestInventoryRecordInterruptible:
    """Verify the interruptible field is set correctly.

    Regression coverage for the bug documented in docs/crucible-config-hierarchy.md §3:
    RunPod's create-pod GraphQL response does not reliably echo back
    the interruptible flag, so the record would fall back to
    previous.get("interruptible", True) — making yaml edits to
    pod.interruptible appear to not stick. The fix added
    requested_interruptible=... to let the provisioner pass the
    authoritative input value.
    """

    def test_requested_interruptible_wins_over_api_echo(self):
        """requested_interruptible=False beats a missing-or-True raw value."""
        raw = {"id": "pod_123"}  # raw has no 'interruptible' field
        rec = inventory_record_from_api(raw, requested_interruptible=False)
        assert rec["interruptible"] is False

    def test_requested_interruptible_wins_over_stale_previous(self):
        """requested_interruptible=False beats a stale previous=True record."""
        raw = {"id": "pod_123"}
        previous = {"interruptible": True}  # wrong previous value
        rec = inventory_record_from_api(
            raw, previous=previous, requested_interruptible=False
        )
        assert rec["interruptible"] is False

    def test_requested_interruptible_true_wins(self):
        """requested_interruptible=True beats a stale previous=False record."""
        raw = {"id": "pod_123"}
        previous = {"interruptible": False}
        rec = inventory_record_from_api(
            raw, previous=previous, requested_interruptible=True
        )
        assert rec["interruptible"] is True

    def test_no_request_falls_back_to_api_then_previous_then_true(self):
        """Without requested_interruptible, legacy fallback chain still works."""
        # Raw has the value -> use raw
        raw = {"id": "pod_123", "interruptible": False}
        rec = inventory_record_from_api(raw)
        assert rec["interruptible"] is False
        # Raw missing, previous has value -> use previous
        raw = {"id": "pod_123"}
        previous = {"interruptible": False}
        rec = inventory_record_from_api(raw, previous=previous)
        assert rec["interruptible"] is False
        # Both missing -> default True
        raw = {"id": "pod_123"}
        rec = inventory_record_from_api(raw)
        assert rec["interruptible"] is True


# ---------------------------------------------------------------------------
# Helpers for stop/start tests
# ---------------------------------------------------------------------------

_MODULE = "crucible.fleet.providers.runpod"


def _make_provider(monkeypatch) -> RunPodProvider:
    """Create a RunPodProvider with minimal config, bypassing API key check."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-xxx")
    return RunPodProvider(
        public_key_path="/dev/null",  # won't be read during stop/start
    )


def _node(
    name: str,
    pod_id: str | None = "pod-abc",
    state: str = "ready",
    *,
    interruptible: bool = False,
    cost_per_hr: float = 0.40,
    gpu_count: int = 1,
) -> dict:
    node: dict = {
        "name": name,
        "state": state,
        "interruptible": interruptible,
        "cost_per_hr": cost_per_hr,
        "gpu_count": gpu_count,
    }
    if pod_id is not None:
        node["pod_id"] = pod_id
    return node


# ---------------------------------------------------------------------------
# TestRunPodProviderStop
# ---------------------------------------------------------------------------


class TestRunPodProviderStop:
    def test_stop_calls_graphql_per_node(self, monkeypatch):
        """runpod_stop_pod is called once for each node."""
        provider = _make_provider(monkeypatch)
        nodes = [
            _node("n1", pod_id="pod-1"),
            _node("n2", pod_id="pod-2"),
            _node("n3", pod_id="pod-3"),
        ]
        with patch(f"{_MODULE}.runpod_stop_pod") as mock_stop:
            provider.stop(nodes)
            assert mock_stop.call_count == 3
            mock_stop.assert_any_call("pod-1")
            mock_stop.assert_any_call("pod-2")
            mock_stop.assert_any_call("pod-3")

    def test_stop_sets_state_to_stopped(self, monkeypatch):
        """Stopped nodes get state='stopped' and api_state='stopped'."""
        provider = _make_provider(monkeypatch)
        nodes = [_node("n1", pod_id="pod-1")]
        with patch(f"{_MODULE}.runpod_stop_pod"):
            result = provider.stop(nodes)
        assert len(result) == 1
        assert result[0]["state"] == "stopped"
        assert result[0]["api_state"] == "stopped"

    def test_stop_skips_unselected_nodes(self, monkeypatch):
        """Only nodes named in selected_names are stopped."""
        provider = _make_provider(monkeypatch)
        nodes = [
            _node("n1", pod_id="pod-1"),
            _node("n2", pod_id="pod-2"),
            _node("n3", pod_id="pod-3"),
        ]
        with patch(f"{_MODULE}.runpod_stop_pod") as mock_stop:
            result = provider.stop(nodes, selected_names={"n1", "n3"})
            assert mock_stop.call_count == 2
            mock_stop.assert_any_call("pod-1")
            mock_stop.assert_any_call("pod-3")
        # n2 should be returned unchanged (still 'ready')
        by_name = {n["name"]: n for n in result}
        assert by_name["n2"]["state"] == "ready"
        assert by_name["n1"]["state"] == "stopped"
        assert by_name["n3"]["state"] == "stopped"

    def test_stop_swallows_fleet_error(self, monkeypatch):
        """If runpod_stop_pod raises FleetError, the node is still returned (best-effort)."""
        provider = _make_provider(monkeypatch)
        nodes = [_node("n1", pod_id="pod-1")]
        with patch(f"{_MODULE}.runpod_stop_pod", side_effect=FleetError("timeout")):
            result = provider.stop(nodes)
        assert len(result) == 1
        # State should be unchanged because the mutation failed
        assert result[0]["name"] == "n1"

    def test_stop_skips_node_without_pod_id(self, monkeypatch):
        """Nodes missing pod_id are passed through unchanged."""
        provider = _make_provider(monkeypatch)
        nodes = [_node("n1", pod_id=None, state="pending")]
        with patch(f"{_MODULE}.runpod_stop_pod") as mock_stop:
            result = provider.stop(nodes)
            mock_stop.assert_not_called()
        assert len(result) == 1
        assert result[0]["state"] == "pending"


# ---------------------------------------------------------------------------
# TestRunPodProviderStart
# ---------------------------------------------------------------------------


class TestRunPodProviderStart:
    def test_start_calls_on_demand_for_non_interruptible(self, monkeypatch):
        """Non-interruptible pods use runpod_start_pod (on-demand resume)."""
        provider = _make_provider(monkeypatch)
        nodes = [_node("n1", pod_id="pod-1", state="stopped", interruptible=False)]
        with (
            patch(f"{_MODULE}.runpod_start_pod") as mock_start,
            patch(f"{_MODULE}.runpod_start_spot_pod") as mock_spot,
            patch.object(provider, "wait_ready", side_effect=lambda ns, **kw: ns),
        ):
            provider.start(nodes)
            mock_start.assert_called_once_with("pod-1")
            mock_spot.assert_not_called()

    def test_start_calls_spot_for_interruptible_with_bid_floor(self, monkeypatch):
        """Interruptible pods use runpod_start_spot_pod with bid = max(cost, 0.10)."""
        provider = _make_provider(monkeypatch)
        nodes = [
            _node(
                "n1",
                pod_id="pod-1",
                state="stopped",
                interruptible=True,
                cost_per_hr=0.40,
                gpu_count=2,
            ),
        ]
        with (
            patch(f"{_MODULE}.runpod_start_pod") as mock_start,
            patch(f"{_MODULE}.runpod_start_spot_pod") as mock_spot,
            patch.object(provider, "wait_ready", side_effect=lambda ns, **kw: ns),
        ):
            provider.start(nodes)
            mock_start.assert_not_called()
            mock_spot.assert_called_once_with("pod-1", bid_per_gpu=0.40, gpu_count=2)

    def test_start_bid_floor_when_cost_is_zero(self, monkeypatch):
        """cost_per_hr=0.0 should bid 0.10 (the floor)."""
        provider = _make_provider(monkeypatch)
        nodes = [
            _node(
                "n1",
                pod_id="pod-1",
                state="stopped",
                interruptible=True,
                cost_per_hr=0.0,
                gpu_count=1,
            ),
        ]
        with (
            patch(f"{_MODULE}.runpod_start_pod"),
            patch(f"{_MODULE}.runpod_start_spot_pod") as mock_spot,
            patch.object(provider, "wait_ready", side_effect=lambda ns, **kw: ns),
        ):
            provider.start(nodes)
            mock_spot.assert_called_once_with("pod-1", bid_per_gpu=0.10, gpu_count=1)

    def test_start_preserves_ordering(self, monkeypatch):
        """Returned list preserves original node order [a, b, c]."""
        provider = _make_provider(monkeypatch)
        node_a = _node("a", pod_id="pod-a", state="stopped")
        node_b = _node("b", pod_id="pod-b", state="ready")
        node_c = _node("c", pod_id="pod-c", state="stopped")
        nodes = [node_a, node_b, node_c]

        def fake_wait_ready(ns, **kw):
            for n in ns:
                n["state"] = "ready"
            return ns

        with (
            patch(f"{_MODULE}.runpod_start_pod"),
            patch.object(provider, "wait_ready", side_effect=fake_wait_ready),
        ):
            result = provider.start(nodes)
        assert [n["name"] for n in result] == ["a", "b", "c"]

    def test_start_sets_state_to_starting(self, monkeypatch):
        """After the GraphQL call but before wait_ready, state is 'starting'."""
        provider = _make_provider(monkeypatch)
        nodes = [_node("n1", pod_id="pod-1", state="stopped")]

        captured_states: list[str] = []

        def spy_wait_ready(ns, **kw):
            # Capture state as seen when wait_ready is invoked
            for n in ns:
                captured_states.append(n["state"])
            return ns

        with (
            patch(f"{_MODULE}.runpod_start_pod"),
            patch.object(provider, "wait_ready", side_effect=spy_wait_ready),
        ):
            provider.start(nodes)
        assert captured_states == ["starting"]
