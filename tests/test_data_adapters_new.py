"""Tests for non-LM data adapters: synthetic images, synthetic video, image folder."""
import pytest
import torch

from crucible.training.data_adapters import (
    DATA_ADAPTER_REGISTRY,
    SyntheticImageAdapter,
    SyntheticVideoAdapter,
    build_data_adapter,
)


class TestSyntheticImageAdapter:
    def test_batch_shape(self):
        adapter = SyntheticImageAdapter(channels=3, num_classes=10)
        batch = adapter.next_batch(batch_size=4, image_size=16)
        assert batch["images"].shape == (4, 3, 16, 16)
        assert batch["labels"].shape == (4,)

    def test_single_channel(self):
        adapter = SyntheticImageAdapter(channels=1, num_classes=5)
        batch = adapter.next_batch(batch_size=2, image_size=8)
        assert batch["images"].shape == (2, 1, 8, 8)

    def test_labels_in_range(self):
        adapter = SyntheticImageAdapter(channels=3, num_classes=7)
        batch = adapter.next_batch(batch_size=100)
        assert batch["labels"].min() >= 0
        assert batch["labels"].max() < 7

    def test_device_placement(self):
        adapter = SyntheticImageAdapter()
        batch = adapter.next_batch(batch_size=2, device="cpu")
        assert batch["images"].device.type == "cpu"

    def test_modality(self):
        assert SyntheticImageAdapter.modality() == "vision"

    def test_registered(self):
        assert "synthetic_images" in DATA_ADAPTER_REGISTRY


class TestSyntheticVideoAdapter:
    def test_batch_shape(self):
        adapter = SyntheticVideoAdapter(num_objects=2, object_size=3)
        batch = adapter.next_batch(batch_size=2, num_frames=4, image_size=16)
        assert batch["frames"].shape == (2, 4, 3, 16, 16)
        assert batch["actions"].shape == (2, 3, 2)  # T-1 actions

    def test_single_frame(self):
        adapter = SyntheticVideoAdapter()
        batch = adapter.next_batch(batch_size=1, num_frames=1, image_size=8)
        assert batch["frames"].shape == (1, 1, 3, 8, 8)
        assert batch["actions"].shape == (1, 0, 2)

    def test_frames_have_content(self):
        adapter = SyntheticVideoAdapter(num_objects=3, object_size=4)
        batch = adapter.next_batch(batch_size=1, num_frames=2, image_size=32)
        # With 3 objects, frames should not be all zeros
        assert batch["frames"].abs().sum() > 0

    def test_device_placement(self):
        adapter = SyntheticVideoAdapter()
        batch = adapter.next_batch(batch_size=1, num_frames=2, image_size=8, device="cpu")
        assert batch["frames"].device.type == "cpu"
        assert batch["actions"].device.type == "cpu"

    def test_modality(self):
        assert SyntheticVideoAdapter.modality() == "world_model"

    def test_registered(self):
        assert "synthetic_video" in DATA_ADAPTER_REGISTRY


class TestBuildDataAdapter:
    def test_build_synthetic_images(self):
        adapter = build_data_adapter("synthetic_images")
        assert isinstance(adapter, SyntheticImageAdapter)

    def test_build_synthetic_video(self):
        adapter = build_data_adapter("synthetic_video")
        assert isinstance(adapter, SyntheticVideoAdapter)

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown data adapter"):
            build_data_adapter("nonexistent_adapter")

    def test_registry_has_all_builtins(self):
        expected = {"token", "image_folder", "synthetic_images", "synthetic_video"}
        assert expected.issubset(set(DATA_ADAPTER_REGISTRY.keys()))
