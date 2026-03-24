"""Tests for diffusion and JEPA objectives."""
import pytest
import torch

from crucible.training.objectives import (
    OBJECTIVE_REGISTRY,
    DiffusionLossObjective,
    JEPAObjective,
    build_objective,
)


class TestDiffusionLossObjective:
    def test_basic_compute(self):
        obj = DiffusionLossObjective()
        pred = torch.randn(4, 3, 8, 8)
        noise = torch.randn(4, 3, 8, 8)
        result = obj.compute({"noise_pred": pred}, {"noise": noise})
        assert "loss" in result
        assert "noise_mse" in result
        assert result["loss"].shape == ()
        assert result["loss"].item() > 0

    def test_perfect_prediction(self):
        obj = DiffusionLossObjective()
        noise = torch.randn(2, 1, 4, 4)
        result = obj.compute({"noise_pred": noise}, {"noise": noise})
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_equals_noise_mse(self):
        obj = DiffusionLossObjective()
        result = obj.compute(
            {"noise_pred": torch.randn(2, 3, 8, 8)},
            {"noise": torch.randn(2, 3, 8, 8)},
        )
        assert result["loss"].item() == pytest.approx(result["noise_mse"].item())

    def test_metric_names(self):
        obj = DiffusionLossObjective()
        assert "noise_mse" in obj.metric_names()

    def test_name(self):
        assert DiffusionLossObjective.name == "diffusion"

    def test_registered(self):
        assert "diffusion" in OBJECTIVE_REGISTRY

    def test_build(self):
        obj = build_objective("diffusion")
        assert isinstance(obj, DiffusionLossObjective)


class TestJEPAObjective:
    def test_basic_compute(self):
        obj = JEPAObjective()
        result = obj.compute(
            {
                "pred_embeddings": torch.randn(4, 64),
                "z_std": torch.randn(64).abs(),
            },
            {"target_embeddings": torch.randn(4, 64)},
        )
        assert "loss" in result
        assert "pred_loss" in result
        assert "var_reg" in result
        assert result["loss"].shape == ()

    def test_perfect_prediction_zero_pred_loss(self):
        obj = JEPAObjective()
        emb = torch.randn(4, 32)
        result = obj.compute(
            {"pred_embeddings": emb, "z_std": torch.ones(32) * 2.0},
            {"target_embeddings": emb},
        )
        assert result["pred_loss"].item() == pytest.approx(0.0, abs=1e-6)
        # var_reg should be 0 since std > target (1.0)
        assert result["var_reg"].item() == pytest.approx(0.0, abs=1e-6)

    def test_var_reg_activates_when_std_low(self):
        obj = JEPAObjective(var_target=1.0)
        result = obj.compute(
            {
                "pred_embeddings": torch.randn(4, 32),
                "z_std": torch.full((32,), 0.1),  # very low std
            },
            {"target_embeddings": torch.randn(4, 32)},
        )
        assert result["var_reg"].item() > 0

    def test_var_weight_scaling(self):
        low_weight = JEPAObjective(var_weight=0.01)
        high_weight = JEPAObjective(var_weight=10.0)
        preds = {
            "pred_embeddings": torch.randn(4, 32),
            "z_std": torch.full((32,), 0.1),
        }
        targets = {"target_embeddings": torch.randn(4, 32)}
        low_result = low_weight.compute(preds, targets)
        high_result = high_weight.compute(preds, targets)
        # Same pred_loss, different total loss
        assert low_result["pred_loss"].item() == pytest.approx(
            high_result["pred_loss"].item(), abs=1e-4
        )
        assert high_result["loss"].item() > low_result["loss"].item()

    def test_metric_names(self):
        obj = JEPAObjective()
        assert "pred_loss" in obj.metric_names()
        assert "var_reg" in obj.metric_names()

    def test_name(self):
        assert JEPAObjective.name == "jepa"

    def test_registered(self):
        assert "jepa" in OBJECTIVE_REGISTRY

    def test_build(self):
        obj = build_objective("jepa")
        assert isinstance(obj, JEPAObjective)
