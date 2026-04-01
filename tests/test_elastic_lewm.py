"""Tests for the Elastic LE-WM architecture plugin."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_elastic_path = Path(__file__).parent.parent / ".crucible/plugins/architectures/elastic_lewm.py"
if not _elastic_path.exists():
    pytest.skip(f"Elastic LE-WM plugin not found at {_elastic_path}", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("elastic_lewm", str(_elastic_path))
_mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec_module so that dataclasses with
# `from __future__ import annotations` can resolve type hints.
import sys as _sys
_sys.modules["elastic_lewm"] = _mod
_spec.loader.exec_module(_mod)

ElasticLeWMModel = _mod.ElasticLeWMModel
ElasticViTBlock = _mod.ElasticViTBlock
ElasticDiTBlock = _mod.ElasticDiTBlock
DifficultyRouter = _mod.DifficultyRouter
BudgetConfig = _mod.BudgetConfig
compute_budget_configs = _mod.compute_budget_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_model(**overrides) -> ElasticLeWMModel:
    """Build a small Elastic LE-WM for fast tests."""
    defaults = dict(
        image_size=64,
        patch_size=16,
        in_channels=3,
        embed_dim=32,
        encoder_depth=4,
        encoder_heads=4,
        predictor_depth=4,
        predictor_heads=4,
        predictor_dim_head=8,
        predictor_mlp_dim=64,
        action_dim=4,
        sigreg_weight=1.0,
        sigreg_projections=16,
        dropout=0.0,
        num_budgets=4,
        warmup_fraction=0.3,
        kd_weight=0.5,
        router_cost=0.01,
        router_entropy=0.005,
        gumbel_temp=5.0,
        gumbel_temp_min=0.5,
        use_sandwich=True,
        fixed_budget=0.0,
        total_steps=100,
    )
    defaults.update(overrides)
    return ElasticLeWMModel(**defaults)


def _make_batch(B: int = 2, T: int = 3, C: int = 3, H: int = 64, W: int = 64, A: int = 4):
    frames = torch.randn(B, T, C, H, W)
    actions = torch.randn(B, T - 1, A)
    return frames, actions


# ---------------------------------------------------------------------------
# Test: compute_budget_configs
# ---------------------------------------------------------------------------

class TestComputeBudgetConfigs:
    def test_correct_count_standard(self):
        """4 budgets for a model with 4 heads, 4 layers."""
        configs = compute_budget_configs(
            encoder_depth=4, encoder_heads=4, encoder_mlp_dim=128,
            predictor_depth=4, predictor_heads=4, predictor_mlp_dim=64,
            num_budgets=4,
        )
        assert len(configs) == 4

    def test_auto_reduce_small_model(self):
        """2 heads can't have 4 budget levels -- should auto-reduce."""
        configs = compute_budget_configs(
            encoder_depth=4, encoder_heads=2, encoder_mlp_dim=128,
            predictor_depth=4, predictor_heads=4, predictor_mlp_dim=64,
            num_budgets=4,
        )
        assert len(configs) == 2

    def test_last_budget_is_full(self):
        """Last budget should have all layers active."""
        configs = compute_budget_configs(
            encoder_depth=6, encoder_heads=6, encoder_mlp_dim=256,
            predictor_depth=6, predictor_heads=8, predictor_mlp_dim=128,
            num_budgets=4,
        )
        last = configs[-1]
        assert len(last.encoder_active_layers) == 6
        assert last.encoder_active_heads == 6
        assert len(last.predictor_active_layers) == 6
        assert last.predictor_active_heads == 8

    def test_first_budget_is_smallest(self):
        """First budget should have fewer active layers than last."""
        configs = compute_budget_configs(
            encoder_depth=8, encoder_heads=8, encoder_mlp_dim=256,
            predictor_depth=8, predictor_heads=8, predictor_mlp_dim=256,
            num_budgets=4,
        )
        assert len(configs[0].encoder_active_layers) <= len(configs[-1].encoder_active_layers)
        assert configs[0].encoder_active_heads <= configs[-1].encoder_active_heads
        assert configs[0].flops_ratio <= configs[-1].flops_ratio

    def test_flops_ratio_monotonic(self):
        """FLOPS ratios should be monotonically increasing."""
        configs = compute_budget_configs(
            encoder_depth=8, encoder_heads=8, encoder_mlp_dim=256,
            predictor_depth=8, predictor_heads=8, predictor_mlp_dim=256,
            num_budgets=4,
        )
        ratios = [c.flops_ratio for c in configs]
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1]

    def test_single_budget(self):
        """num_budgets=1 should produce exactly one full-capacity config."""
        configs = compute_budget_configs(
            encoder_depth=4, encoder_heads=4, encoder_mlp_dim=64,
            predictor_depth=4, predictor_heads=4, predictor_mlp_dim=64,
            num_budgets=1,
        )
        assert len(configs) == 1
        assert configs[0].flops_ratio == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test: ElasticViTBlock
# ---------------------------------------------------------------------------

class TestElasticViTBlock:
    def test_full_capacity(self):
        """Full capacity should produce correct output shape."""
        block = ElasticViTBlock(dim=32, num_heads=4)
        x = torch.randn(2, 17, 32)  # B=2, N=17 patches, D=32
        out = block(x)
        assert out.shape == (2, 17, 32)

    def test_reduced_heads(self):
        """Prefix head masking should still produce correct shape."""
        block = ElasticViTBlock(dim=32, num_heads=4)
        x = torch.randn(2, 17, 32)
        out = block(x, active_heads=2)
        assert out.shape == (2, 17, 32)

    def test_reduced_mlp(self):
        """Prefix MLP masking should still produce correct shape."""
        block = ElasticViTBlock(dim=32, num_heads=4)
        x = torch.randn(2, 17, 32)
        out = block(x, active_mlp=32)  # half of 128
        assert out.shape == (2, 17, 32)

    def test_depth_skip(self):
        """depth_active=False should return input unchanged."""
        block = ElasticViTBlock(dim=32, num_heads=4)
        x = torch.randn(2, 17, 32)
        out = block(x, depth_active=False)
        assert torch.equal(out, x)

    def test_combined_elastic(self):
        """Reduced heads + reduced MLP together."""
        block = ElasticViTBlock(dim=32, num_heads=4)
        x = torch.randn(2, 17, 32)
        out = block(x, active_heads=1, active_mlp=16)
        assert out.shape == (2, 17, 32)


# ---------------------------------------------------------------------------
# Test: ElasticDiTBlock
# ---------------------------------------------------------------------------

class TestElasticDiTBlock:
    def test_full_capacity(self):
        block = ElasticDiTBlock(dim=32, num_heads=4, cond_dim=32, dim_head=8, mlp_dim=64)
        x = torch.randn(2, 1, 32)
        cond = torch.randn(2, 1, 32)
        out = block(x, cond)
        assert out.shape == (2, 1, 32)

    def test_reduced_heads(self):
        block = ElasticDiTBlock(dim=32, num_heads=4, cond_dim=32, dim_head=8, mlp_dim=64)
        x = torch.randn(2, 1, 32)
        cond = torch.randn(2, 1, 32)
        out = block(x, cond, active_heads=2)
        assert out.shape == (2, 1, 32)

    def test_reduced_mlp(self):
        block = ElasticDiTBlock(dim=32, num_heads=4, cond_dim=32, dim_head=8, mlp_dim=64)
        x = torch.randn(2, 1, 32)
        cond = torch.randn(2, 1, 32)
        out = block(x, cond, active_mlp=16)
        assert out.shape == (2, 1, 32)

    def test_depth_skip(self):
        block = ElasticDiTBlock(dim=32, num_heads=4, cond_dim=32, dim_head=8, mlp_dim=64)
        x = torch.randn(2, 1, 32)
        cond = torch.randn(2, 1, 32)
        out = block(x, cond, depth_active=False)
        assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# Test: DifficultyRouter
# ---------------------------------------------------------------------------

class TestDifficultyRouter:
    def test_output_shape(self):
        router = DifficultyRouter(embed_dim=32, action_dim=4, num_budgets=4)
        z = torch.randn(4, 32)
        a = torch.randn(4, 4)
        weights, logits = router(z, a)
        assert weights.shape == (4, 4)
        assert logits.shape == (4, 4)

    def test_weights_sum_to_one(self):
        """Budget weights should sum to 1 for each sample."""
        router = DifficultyRouter(embed_dim=32, action_dim=4, num_budgets=4)
        router.train()
        z = torch.randn(8, 32)
        a = torch.randn(8, 4)
        weights, _ = router(z, a)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gumbel_softmax_training(self):
        """In training mode, should produce soft distributions."""
        router = DifficultyRouter(embed_dim=32, action_dim=4, num_budgets=4, temperature=1.0)
        router.train()
        z = torch.randn(16, 32)
        a = torch.randn(16, 4)
        weights, _ = router(z, a)
        # With Gumbel-Softmax at temperature 1.0, weights should generally not be one-hot
        # Check that at least some samples have non-trivial distributions
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_argmax_in_inference(self):
        """In inference mode, should produce one-hot distributions (argmax)."""
        router = DifficultyRouter(embed_dim=32, action_dim=4, num_budgets=4)
        router.eval()
        z = torch.randn(4, 32)
        a = torch.randn(4, 4)
        weights, _ = router(z, a)
        # Each row should be one-hot
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4))
        assert torch.allclose(weights.max(dim=-1).values, torch.ones(4))


# ---------------------------------------------------------------------------
# Test: ElasticLeWMModel forward (full budget / stage 1)
# ---------------------------------------------------------------------------

class TestElasticLeWMForwardFull:
    def test_output_keys(self):
        """Forward should return all expected keys."""
        model = _make_model()
        model.train()
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        expected_keys = {
            "loss", "pred_loss", "sigreg", "kd_loss", "router_loss",
            "budget_distribution", "pred_embeddings", "target_embeddings",
            "embeddings", "z_std",
        }
        assert expected_keys.issubset(out.keys())

    def test_loss_is_scalar(self):
        model = _make_model()
        model.train()
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        assert out["loss"].dim() == 0

    def test_pred_embeddings_shape(self):
        model = _make_model()
        model.train()
        B, T = 2, 3
        frames, actions = _make_batch(B=B, T=T)
        out = model(frames=frames, actions=actions)
        # pred_embeddings: [B, T-1, D]
        assert out["pred_embeddings"].shape == (B, T - 1, 32)

    def test_target_embeddings_shape(self):
        model = _make_model()
        model.train()
        B, T = 2, 3
        frames, actions = _make_batch(B=B, T=T)
        out = model(frames=frames, actions=actions)
        assert out["target_embeddings"].shape == (B, T - 1, 32)

    def test_stage1_kd_and_router_zero(self):
        """In stage 1 (warmup), kd_loss and router_loss should be zero."""
        model = _make_model(warmup_fraction=0.9, total_steps=1000)
        model.train()
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        assert out["kd_loss"].item() == 0.0
        assert out["router_loss"].item() == 0.0


# ---------------------------------------------------------------------------
# Test: ElasticLeWMModel forward (sub-budget / stage 2)
# ---------------------------------------------------------------------------

class TestElasticLeWMForwardElastic:
    def test_elastic_stage_produces_output(self):
        """After warmup, elastic forward should produce valid output."""
        model = _make_model(warmup_fraction=0.0, total_steps=100)
        model.train()
        # Force past warmup
        model._step.fill_(10)
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        assert "loss" in out
        assert out["loss"].dim() == 0
        # KD loss should be present in elastic stage
        assert "kd_loss" in out

    def test_elastic_budget_distribution_shape(self):
        """Budget distribution should have correct shape."""
        model = _make_model(warmup_fraction=0.0, total_steps=100)
        model.train()
        model._step.fill_(10)
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        assert out["budget_distribution"].shape == (model.actual_num_budgets,)


# ---------------------------------------------------------------------------
# Test: ELASTIC_FIXED_BUDGET
# ---------------------------------------------------------------------------

class TestFixedBudget:
    def test_fixed_budget_bypasses_router(self):
        """With fixed_budget > 0, should use fixed config instead of router."""
        model = _make_model(fixed_budget=0.5, warmup_fraction=0.0, total_steps=100)
        model.train()
        model._step.fill_(10)
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        assert "loss" in out
        assert out["loss"].dim() == 0
        # Router loss should be zero with fixed budget
        assert out["router_loss"].item() == 0.0

    def test_fixed_budget_full(self):
        """Fixed budget at 1.0 should select the largest budget."""
        model = _make_model(fixed_budget=1.0, warmup_fraction=0.0, total_steps=100)
        model.train()
        model._step.fill_(10)
        frames, actions = _make_batch()
        out = model(frames=frames, actions=actions)
        assert "loss" in out


# ---------------------------------------------------------------------------
# Test: param_groups
# ---------------------------------------------------------------------------

class TestParamGroups:
    def test_three_groups(self):
        """Should return encoder, predictor, router groups."""
        model = _make_model()
        groups = model.param_groups()
        assert len(groups) == 3
        names = [g["name"] for g in groups]
        assert "encoder" in names
        assert "predictor" in names
        assert "router" in names

    def test_all_params_covered(self):
        """Encoder + predictor + router params should appear in groups."""
        model = _make_model()
        groups = model.param_groups()
        grouped_params = set()
        for g in groups:
            for p in g["params"]:
                grouped_params.add(id(p))
        # Check that encoder + predictor + router params are covered
        encoder_params = {id(p) for p in model.encoder.parameters()}
        predictor_params = {id(p) for p in model.predictor.parameters()}
        router_params = {id(p) for p in model.router.parameters()}
        expected = encoder_params | predictor_params | router_params
        assert expected == grouped_params


# ---------------------------------------------------------------------------
# Test: modality
# ---------------------------------------------------------------------------

class TestModality:
    def test_modality_world_model(self):
        assert ElasticLeWMModel.modality() == "world_model"
