"""Tests for crucible.training.objectives."""
from __future__ import annotations

import pytest
import torch

from crucible.training.objectives import (
    OBJECTIVE_REGISTRY,
    CompositeObjective,
    CrossEntropyObjective,
    KLDivergenceObjective,
    MSEObjective,
    TrainingObjective,
    build_objective,
    register_objective,
)


# ---------------------------------------------------------------------------
# CrossEntropyObjective
# ---------------------------------------------------------------------------


class TestCrossEntropyObjective:
    def test_computes_loss(self):
        obj = CrossEntropyObjective()
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        result = obj.compute({"logits": logits}, {"labels": labels})
        assert "loss" in result
        assert result["loss"].dim() == 0
        assert result["loss"].item() > 0

    def test_3d_logits(self):
        """Cross-entropy with (batch, seq, vocab) logits should flatten automatically."""
        obj = CrossEntropyObjective()
        logits = torch.randn(2, 5, 10)
        labels = torch.randint(0, 10, (2, 5))
        result = obj.compute({"logits": logits}, {"labels": labels})
        assert result["loss"].dim() == 0

    def test_ignore_index(self):
        obj = CrossEntropyObjective(ignore_index=-1)
        logits = torch.randn(4, 10)
        labels = torch.tensor([0, 1, -1, 3])
        result = obj.compute({"logits": logits}, {"labels": labels})
        assert result["loss"].dim() == 0

    def test_name(self):
        assert CrossEntropyObjective.name == "cross_entropy"


# ---------------------------------------------------------------------------
# MSEObjective
# ---------------------------------------------------------------------------


class TestMSEObjective:
    def test_computes_loss(self):
        obj = MSEObjective()
        output = torch.randn(4, 3)
        target = torch.randn(4, 3)
        result = obj.compute({"output": output}, {"target": target})
        assert "loss" in result
        assert result["loss"].dim() == 0
        assert result["loss"].item() >= 0

    def test_zero_loss_when_equal(self):
        obj = MSEObjective()
        x = torch.tensor([1.0, 2.0, 3.0])
        result = obj.compute({"output": x}, {"target": x.clone()})
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_name(self):
        assert MSEObjective.name == "mse"


# ---------------------------------------------------------------------------
# KLDivergenceObjective
# ---------------------------------------------------------------------------


class TestKLDivergenceObjective:
    def test_computes_loss(self):
        obj = KLDivergenceObjective()
        log_probs = torch.log_softmax(torch.randn(4, 10), dim=-1)
        target_probs = torch.softmax(torch.randn(4, 10), dim=-1)
        result = obj.compute(
            {"log_probs": log_probs}, {"target_probs": target_probs}
        )
        assert "loss" in result
        assert result["loss"].dim() == 0

    def test_name(self):
        assert KLDivergenceObjective.name == "kl_divergence"


# ---------------------------------------------------------------------------
# CompositeObjective
# ---------------------------------------------------------------------------


class TestCompositeObjective:
    def test_combines_losses(self):
        ce = CrossEntropyObjective()
        mse = MSEObjective()
        composite = CompositeObjective([(1.0, ce), (0.5, mse)])

        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        output = torch.randn(4, 10)
        target = torch.randn(4, 10)

        predictions = {"logits": logits, "output": output}
        targets = {"labels": labels, "target": target}
        result = composite.compute(predictions, targets)

        assert "loss" in result
        assert "loss_cross_entropy" in result
        assert "loss_mse" in result

    def test_weighted_sum(self):
        """Total loss should be the weighted sum of sub-losses."""
        mse1 = MSEObjective()
        mse1.name = "mse_a"
        mse2 = MSEObjective()
        mse2.name = "mse_b"
        composite = CompositeObjective([(2.0, mse1), (3.0, mse2)])

        output = torch.tensor([1.0])
        target = torch.tensor([0.0])
        predictions = {"output": output}
        targets = {"target": target}
        result = composite.compute(predictions, targets)

        # Each MSE loss is 1.0, so total = 2*1 + 3*1 = 5.0
        assert result["loss"].item() == pytest.approx(5.0, abs=1e-5)

    def test_empty_objectives_raises(self):
        with pytest.raises(ValueError):
            CompositeObjective([])

    def test_metric_names(self):
        ce = CrossEntropyObjective()
        mse = MSEObjective()
        composite = CompositeObjective([(1.0, ce), (0.5, mse)])
        names = composite.metric_names()
        assert "loss_cross_entropy" in names
        assert "loss_mse" in names


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtins_registered(self):
        assert "cross_entropy" in OBJECTIVE_REGISTRY
        assert "mse" in OBJECTIVE_REGISTRY
        assert "kl_divergence" in OBJECTIVE_REGISTRY
        assert "composite" in OBJECTIVE_REGISTRY

    def test_build_objective(self):
        obj = build_objective("cross_entropy")
        assert isinstance(obj, CrossEntropyObjective)

    def test_build_mse(self):
        obj = build_objective("mse")
        assert isinstance(obj, MSEObjective)

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown objective"):
            build_objective("nonexistent_loss")

    def test_register_custom(self):
        class CustomObjective(TrainingObjective):
            name = "custom_test"

            def compute(self, predictions, targets):
                return {"loss": torch.tensor(0.0)}

        register_objective("custom_test", CustomObjective)
        assert "custom_test" in OBJECTIVE_REGISTRY
        obj = build_objective("custom_test")
        assert isinstance(obj, CustomObjective)
        # Clean up
        del OBJECTIVE_REGISTRY["custom_test"]
