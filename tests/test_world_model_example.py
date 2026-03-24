"""Smoke tests for the world model example."""
import sys
from pathlib import Path

import pytest
import torch

# Add repo root to path so examples/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestJEPAWorldModel:
    def test_forward_pass(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=32, action_dim=2, base_channels=8
        )
        frames = torch.randn(2, 4, 3, 16, 16)
        actions = torch.randn(2, 3, 2)
        result = model(frames=frames, actions=actions)
        assert "loss" in result
        assert "pred_loss" in result
        assert "var_reg" in result
        assert result["loss"].shape == ()
        assert result["loss"].item() > 0

    def test_training_step(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=32, action_dim=2, base_channels=8
        )
        frames = torch.randn(2, 3, 3, 16, 16)
        actions = torch.randn(2, 2, 2)
        result = model.training_step(frames=frames, actions=actions)
        assert "loss" in result
        # Loss should be differentiable (only online encoder + predictor)
        result["loss"].backward()
        grad_count = sum(1 for p in model.encoder.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_ema_update(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=16, action_dim=2, base_channels=4,
            ema_decay=0.5,
        )
        # Manually perturb online encoder so it differs from target
        with torch.no_grad():
            model.encoder.proj.weight.add_(torch.ones_like(model.encoder.proj.weight))

        target_before = model.target_encoder.proj.weight.clone()

        # Run a forward pass (triggers EMA update)
        frames = torch.randn(2, 3, 3, 8, 8)
        actions = torch.randn(2, 2, 2)
        model(frames=frames, actions=actions)

        target_after = model.target_encoder.proj.weight
        # Target should have changed due to EMA (online != target now)
        assert not torch.allclose(target_before, target_after)

    def test_target_encoder_no_grad(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=16, action_dim=2, base_channels=4
        )
        for p in model.target_encoder.parameters():
            assert not p.requires_grad

    def test_encode(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=32, action_dim=2, base_channels=8
        )
        # Single frame batch
        frames = torch.randn(2, 3, 16, 16)
        z = model.encode(frames)
        assert z.shape == (2, 32)

        # Multi-frame batch
        frames_seq = torch.randn(2, 5, 3, 16, 16)
        z_seq = model.encode(frames_seq)
        assert z_seq.shape == (2, 5, 32)

    def test_predict_next(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=32, action_dim=2, base_channels=8
        )
        z = torch.randn(2, 32)
        action = torch.randn(2, 2)
        z_next = model.predict_next(z, action)
        assert z_next.shape == (2, 32)

    def test_modality(self):
        from examples.world_model.model import JEPAWorldModel

        assert JEPAWorldModel.modality() == "world_model"

    def test_factory_registration(self):
        from examples.world_model.model import register
        from crucible.models.registry import _REGISTRY as MODEL_REGISTRY

        try:
            register()
        except ValueError:
            pass  # Already registered from earlier test imports
        assert "jepa_wm" in MODEL_REGISTRY

    def test_validation_step(self):
        from examples.world_model.model import JEPAWorldModel

        model = JEPAWorldModel(
            in_channels=3, embed_dim=16, action_dim=2, base_channels=4
        )
        frames = torch.randn(2, 3, 3, 8, 8)
        actions = torch.randn(2, 2, 2)
        result = model.validation_step(frames=frames, actions=actions)
        assert "loss" in result
        assert "pred_loss" in result


class TestBouncingBallAdapter:
    def test_batch_shape(self):
        from examples.world_model.data_adapter import BouncingBallAdapter

        adapter = BouncingBallAdapter()
        batch = adapter.next_batch(batch_size=2, num_frames=4, image_size=16)
        assert batch["frames"].shape == (2, 4, 3, 16, 16)
        assert batch["actions"].shape == (2, 3, 2)

    def test_frames_normalized(self):
        from examples.world_model.data_adapter import BouncingBallAdapter

        adapter = BouncingBallAdapter()
        batch = adapter.next_batch(batch_size=1, num_frames=2, image_size=16)
        # After normalization to [-1, 1]
        assert batch["frames"].min() >= -1.0
        assert batch["frames"].max() <= 1.0

    def test_modality(self):
        from examples.world_model.data_adapter import BouncingBallAdapter

        assert BouncingBallAdapter.modality() == "world_model"

    def test_registration(self):
        from examples.world_model.data_adapter import register

        register()
        from crucible.training.data_adapters import DATA_ADAPTER_REGISTRY

        assert "bouncing_balls" in DATA_ADAPTER_REGISTRY
