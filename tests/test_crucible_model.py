"""Tests for CrucibleModel abstract base class and TiedEmbeddingLM integration."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from crucible.models.base import CrucibleModel, TiedEmbeddingLM


# ---------------------------------------------------------------------------
# CrucibleModel is ABC
# ---------------------------------------------------------------------------


class TestCrucibleModelABC:
    def test_cannot_instantiate_directly(self):
        """CrucibleModel should not be instantiable because forward() is abstract."""
        with pytest.raises(TypeError):
            CrucibleModel()

    def test_concrete_subclass_works(self):
        """A subclass that implements forward() can be instantiated."""

        class DummyModel(CrucibleModel):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, **batch) -> dict[str, Tensor]:
                x = batch["x"]
                return {"loss": self.linear(x).sum()}

        model = DummyModel()
        result = model.forward(x=torch.randn(1, 4))
        assert "loss" in result

    def test_default_training_step_delegates_to_forward(self):
        """Default training_step should call forward."""

        class DummyModel(CrucibleModel):
            def __init__(self):
                super().__init__()

            def forward(self, **batch) -> dict[str, Tensor]:
                return {"loss": torch.tensor(42.0)}

        model = DummyModel()
        result = model.training_step()
        assert result["loss"].item() == 42.0

    def test_default_validation_step_delegates_to_forward(self):
        """Default validation_step should call forward."""

        class DummyModel(CrucibleModel):
            def __init__(self):
                super().__init__()

            def forward(self, **batch) -> dict[str, Tensor]:
                return {"loss": torch.tensor(7.0)}

        model = DummyModel()
        result = model.validation_step()
        assert result["loss"].item() == 7.0

    def test_default_metric_names(self):
        class DummyModel(CrucibleModel):
            def __init__(self):
                super().__init__()

            def forward(self, **batch) -> dict[str, Tensor]:
                return {"loss": torch.tensor(0.0)}

        model = DummyModel()
        assert model.metric_names() == []

    def test_default_param_groups(self):
        class DummyModel(CrucibleModel):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(3))

            def forward(self, **batch) -> dict[str, Tensor]:
                return {"loss": self.w.sum()}

        model = DummyModel()
        groups = model.param_groups()
        assert len(groups) == 1
        assert len(groups[0]["params"]) == 1

    def test_default_modality(self):
        class DummyModel(CrucibleModel):
            def __init__(self):
                super().__init__()

            def forward(self, **batch) -> dict[str, Tensor]:
                return {"loss": torch.tensor(0.0)}

        assert DummyModel.modality() == "generic"


# ---------------------------------------------------------------------------
# TiedEmbeddingLM inherits from CrucibleModel
# ---------------------------------------------------------------------------


class TestTiedEmbeddingLMInheritance:
    def test_isinstance_crucible_model(self):
        """TiedEmbeddingLM should be a CrucibleModel."""
        assert issubclass(TiedEmbeddingLM, CrucibleModel)

    def test_modality_returns_lm(self):
        """TiedEmbeddingLM.modality() should return 'lm'."""
        assert TiedEmbeddingLM.modality() == "lm"


# ---------------------------------------------------------------------------
# TiedEmbeddingLM.training_step bridge
# ---------------------------------------------------------------------------


class _MinimalLM(TiedEmbeddingLM):
    """Minimal concrete subclass for testing the training_step bridge."""

    def __init__(self):
        super().__init__(
            vocab_size=100,
            model_dim=32,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
        )
        # Minimal "hidden" layer
        self._proj = nn.Linear(32, 32)

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        return self._proj(self.embed_tokens(input_ids))


class TestTrainingStepBridge:
    def test_training_step_returns_loss_dict(self):
        model = _MinimalLM()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "target_ids": torch.randint(0, 100, (2, 8)),
        }
        result = model.training_step(**batch)
        assert "loss" in result
        assert result["loss"].dim() == 0  # scalar

    def test_training_step_loss_matches_forward(self):
        model = _MinimalLM()
        x = torch.randint(0, 100, (2, 8))
        y = torch.randint(0, 100, (2, 8))
        forward_loss = model(x, y)
        step_loss = model.training_step(input_ids=x, target_ids=y)["loss"]
        assert torch.allclose(forward_loss, step_loss)

    def test_forward_positional_args_still_work(self):
        """The existing forward(input_ids, target_ids) calling convention must work."""
        model = _MinimalLM()
        x = torch.randint(0, 100, (2, 8))
        y = torch.randint(0, 100, (2, 8))
        # Positional call -- must not break
        loss = model(x, y)
        assert loss.dim() == 0

    def test_training_step_with_lora_none(self):
        model = _MinimalLM()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "target_ids": torch.randint(0, 100, (2, 8)),
            "lora": None,
        }
        result = model.training_step(**batch)
        assert "loss" in result
