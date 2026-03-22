"""Tests for built-in and plugin architectures — forward pass smoke tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from crucible.models.registry import build_model, list_families


@pytest.fixture
def base_args() -> SimpleNamespace:
    """Minimal args namespace for architecture instantiation."""
    return SimpleNamespace(
        vocab_size=256,
        model_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        mlp_mult=2,
        rope_base=10000.0,
        qk_gain_init=1.0,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        embed_bottleneck_dim=0,
        spectral_embed_init=False,
        attention_variant="standard",
        residual_variant="standard",
        activation="relu_sq",
        model_family="baseline",
        # Looped/convloop specific
        recurrence_steps=4,
        share_blocks=2,
        state_dim=64,
        # Baseline extras
        block_pattern="",
        use_smear_gate=False,
        use_bigram_hash=False,
        bigram_hash_buckets=2048,
        bigram_hash_embed_dim=128,
        ortho_init=False,
        use_conv_block=False,
        conv_kernel=3,
        multiscale_window=0,
        token_merge_layer=0,
        token_merge_threshold=0.9,
        use_trigram_hash=False,
        trigram_hash_buckets=4096,
        use_moe=False,
        moe_num_experts=4,
        moe_top_k=2,
        smear_gate=False,
        bigram_hash=False,
        conv_block=False,
        trigram_hash=False,
    )


def _forward_check(model, seq_len: int = 16):
    """Run a forward pass and verify the output is a scalar loss."""
    ids = torch.randint(0, 256, (1, seq_len))
    loss = model(ids, ids)
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss is not finite"


class TestBuiltinArchitectures:
    def test_baseline_forward(self, base_args):
        base_args.model_family = "baseline"
        model = build_model(base_args)
        _forward_check(model)

    def test_looped_forward(self, base_args):
        base_args.model_family = "looped"
        model = build_model(base_args)
        _forward_check(model)

    def test_convloop_forward(self, base_args):
        base_args.model_family = "convloop"
        model = build_model(base_args)
        _forward_check(model)

    def test_prefix_memory_forward(self, base_args):
        base_args.model_family = "prefix_memory"
        model = build_model(base_args)
        _forward_check(model)


class TestPluginArchitecture:
    def test_two_tower_forward(self, base_args):
        base_args.model_family = "two_tower"
        model = build_model(base_args)
        _forward_check(model)

    def test_two_tower_registered(self):
        families = list_families()
        assert "two_tower" in families


class TestRegistry:
    def test_all_builtins_registered(self):
        families = list_families()
        for name in ["baseline", "looped", "convloop", "prefix_memory"]:
            assert name in families, f"{name} not registered"

    def test_unknown_family_raises(self, base_args):
        base_args.model_family = "nonexistent_family_xyz"
        with pytest.raises(ValueError, match="unsupported MODEL_FAMILY"):
            build_model(base_args)

    def test_missing_model_family_attr(self):
        args = SimpleNamespace()  # no model_family
        with pytest.raises(ValueError, match="model_family"):
            build_model(args)
