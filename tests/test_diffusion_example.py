"""Smoke tests for the diffusion model example."""
import sys
from pathlib import Path

import pytest
import torch

# Add repo root to path so examples/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDDPMModel:
    def test_forward_pass(self):
        from examples.diffusion.model import DDPMModel

        model = DDPMModel(in_channels=1, base_channels=16, diffusion_steps=10)
        images = torch.randn(2, 1, 28, 28)
        result = model(images=images)
        assert "loss" in result
        assert "noise_mse" in result
        assert result["loss"].shape == ()
        assert result["loss"].item() > 0

    def test_training_step(self):
        from examples.diffusion.model import DDPMModel

        model = DDPMModel(in_channels=1, base_channels=16, diffusion_steps=10)
        images = torch.randn(2, 1, 28, 28)
        result = model.training_step(images=images)
        assert "loss" in result
        # Loss should be differentiable
        result["loss"].backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_validation_step(self):
        from examples.diffusion.model import DDPMModel

        model = DDPMModel(in_channels=1, base_channels=16, diffusion_steps=10)
        images = torch.randn(2, 1, 28, 28)
        result = model.validation_step(images=images)
        assert "loss" in result

    def test_modality(self):
        from examples.diffusion.model import DDPMModel

        assert DDPMModel.modality() == "diffusion"

    def test_different_resolutions(self):
        from examples.diffusion.model import DDPMModel

        # 32x32 color images
        model = DDPMModel(in_channels=3, base_channels=16, diffusion_steps=10)
        images = torch.randn(1, 3, 32, 32)
        result = model(images=images)
        assert result["loss"].shape == ()

    def test_factory_registration(self):
        from examples.diffusion.model import register
        from crucible.models.registry import _REGISTRY as MODEL_REGISTRY

        try:
            register()
        except ValueError:
            pass  # Already registered from earlier test imports
        assert "ddpm_unet" in MODEL_REGISTRY

    def test_param_count_reasonable(self):
        from examples.diffusion.model import DDPMModel

        model = DDPMModel(in_channels=1, base_channels=32, diffusion_steps=100)
        params = sum(p.numel() for p in model.parameters())
        # Should be a real model, not trivially small
        assert params > 10_000
        # But not huge — this is a small demo model
        assert params < 50_000_000


class TestDiffusionSchedule:
    def test_q_sample_shape(self):
        from examples.diffusion.model import DiffusionSchedule

        schedule = DiffusionSchedule(timesteps=100)
        x = torch.randn(2, 1, 8, 8)
        t = torch.tensor([10, 50])
        noise = torch.randn_like(x)
        noisy = schedule.q_sample(x, t, noise)
        assert noisy.shape == x.shape

    def test_t0_mostly_signal(self):
        from examples.diffusion.model import DiffusionSchedule

        schedule = DiffusionSchedule(timesteps=1000)
        x = torch.ones(1, 1, 4, 4)
        t = torch.tensor([0])
        noise = torch.zeros_like(x)
        noisy = schedule.q_sample(x, t, noise)
        # At t=0, noisy should be very close to x
        assert (noisy - x).abs().max().item() < 0.1
