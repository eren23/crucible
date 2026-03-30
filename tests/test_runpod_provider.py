"""Tests for RunPod provider payload validation."""

import pytest

from crucible.core.errors import FleetError
from crucible.fleet.providers.runpod import build_pod_payload


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
