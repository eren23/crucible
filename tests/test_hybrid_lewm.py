"""Tests for the Hybrid LE-WM architecture plugin."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_hybrid_path = Path(__file__).parent.parent / ".crucible/plugins/architectures/hybrid_lewm.py"
if not _hybrid_path.exists():
    pytest.skip(f"Hybrid LE-WM plugin not found at {_hybrid_path}", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("hybrid_lewm", str(_hybrid_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BidirectionalLinearAttention = _mod.BidirectionalLinearAttention
LinearViTBlock = _mod.LinearViTBlock
HybridViTEncoder = _mod.HybridViTEncoder
HybridLeWMModel = _mod.HybridLeWMModel
WeightNormProjection = _mod.WeightNormProjection


# ---------------------------------------------------------------------------
# Small defaults for fast tests
# ---------------------------------------------------------------------------
_DIM = 32
_HEADS = 2
_DEPTH = 2
_PATCH = 16
_IMG = 64
_META = 4


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_slim_model(
    block_pattern: str = "ALALAL",
    meta_tokens: int = _META,
) -> HybridLeWMModel:
    """Build a small Hybrid LE-WM for fast unit tests."""
    return HybridLeWMModel(
        image_size=_IMG,
        patch_size=_PATCH,
        in_channels=3,
        embed_dim=_DIM,
        encoder_depth=_DEPTH,
        encoder_heads=_HEADS,
        block_pattern=block_pattern,
        meta_tokens=meta_tokens,
        linear_attn_eps=1e-6,
        predictor_depth=_DEPTH,
        predictor_heads=2,
        predictor_dim_head=16,
        predictor_mlp_dim=64,
        action_dim=4,
        sigreg_weight=1.0,
        sigreg_projections=16,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# BidirectionalLinearAttention
# ---------------------------------------------------------------------------

class TestBidirectionalLinearAttention:
    def test_output_shape(self):
        """Linear attention output should match input shape [B, N, D]."""
        attn = BidirectionalLinearAttention(dim=_DIM, num_heads=_HEADS)
        x = torch.randn(2, 10, _DIM)
        out = attn(x)
        assert out.shape == (2, 10, _DIM)

    def test_single_token(self):
        """Should work with a single token sequence."""
        attn = BidirectionalLinearAttention(dim=_DIM, num_heads=_HEADS)
        x = torch.randn(1, 1, _DIM)
        out = attn(x)
        assert out.shape == (1, 1, _DIM)

    def test_gradient_flow(self):
        """Gradients should flow through linear attention."""
        attn = BidirectionalLinearAttention(dim=_DIM, num_heads=_HEADS)
        x = torch.randn(2, 5, _DIM, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# LinearViTBlock
# ---------------------------------------------------------------------------

class TestLinearViTBlock:
    def test_output_shape(self):
        """LinearViTBlock should preserve input shape."""
        block = LinearViTBlock(dim=_DIM, num_heads=_HEADS)
        x = torch.randn(2, 10, _DIM)
        out = block(x)
        assert out.shape == (2, 10, _DIM)

    def test_residual_connection(self):
        """Output should differ from input (non-identity due to learned params)."""
        block = LinearViTBlock(dim=_DIM, num_heads=_HEADS)
        x = torch.randn(2, 10, _DIM)
        out = block(x)
        # Very unlikely to be exactly equal after random init
        assert not torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# HybridViTEncoder
# ---------------------------------------------------------------------------

class TestHybridViTEncoder:
    def test_alalal_pattern(self):
        """ALALAL pattern should produce [B, embed_dim] output."""
        enc = HybridViTEncoder(
            image_size=_IMG, patch_size=_PATCH, embed_dim=_DIM,
            depth=_DEPTH, num_heads=_HEADS, block_pattern="ALALAL",
            meta_tokens=_META,
        )
        enc.train()
        x = torch.randn(2, 3, _IMG, _IMG)
        out = enc(x)
        assert out.shape == (2, _DIM)

    def test_all_attention_pattern(self):
        """AAAAAA (all softmax) should produce [B, embed_dim] output."""
        enc = HybridViTEncoder(
            image_size=_IMG, patch_size=_PATCH, embed_dim=_DIM,
            depth=_DEPTH, num_heads=_HEADS, block_pattern="AAAAAA",
            meta_tokens=_META,
        )
        enc.train()
        x = torch.randn(2, 3, _IMG, _IMG)
        out = enc(x)
        assert out.shape == (2, _DIM)

    def test_all_linear_pattern(self):
        """LLLLLL (all linear) should produce [B, embed_dim] output."""
        enc = HybridViTEncoder(
            image_size=_IMG, patch_size=_PATCH, embed_dim=_DIM,
            depth=_DEPTH, num_heads=_HEADS, block_pattern="LLLLLL",
            meta_tokens=_META,
        )
        enc.train()
        x = torch.randn(2, 3, _IMG, _IMG)
        out = enc(x)
        assert out.shape == (2, _DIM)

    def test_no_meta_tokens(self):
        """Encoder should work with meta_tokens=0."""
        enc = HybridViTEncoder(
            image_size=_IMG, patch_size=_PATCH, embed_dim=_DIM,
            depth=_DEPTH, num_heads=_HEADS, block_pattern="AL",
            meta_tokens=0,
        )
        enc.train()
        x = torch.randn(2, 3, _IMG, _IMG)
        out = enc(x)
        assert out.shape == (2, _DIM)

    def test_meta_tokens_increase_pos_embed(self):
        """More meta tokens should increase the position embedding size."""
        enc_0 = HybridViTEncoder(
            image_size=_IMG, patch_size=_PATCH, embed_dim=_DIM,
            depth=1, num_heads=_HEADS, block_pattern="A",
            meta_tokens=0,
        )
        enc_4 = HybridViTEncoder(
            image_size=_IMG, patch_size=_PATCH, embed_dim=_DIM,
            depth=1, num_heads=_HEADS, block_pattern="A",
            meta_tokens=4,
        )
        # pos_embed shape[1] should differ by exactly meta_tokens
        assert enc_4.pos_embed.shape[1] == enc_0.pos_embed.shape[1] + 4


# ---------------------------------------------------------------------------
# HybridLeWMModel forward
# ---------------------------------------------------------------------------

class TestHybridLeWMModelForward:
    def test_forward_output_keys(self):
        """Forward should return loss, pred_loss, sigreg, embeddings, etc."""
        model = _make_slim_model()
        model.train()
        B, T = 2, 3
        C, H, W = 3, _IMG, _IMG
        A = 4
        frames = torch.randn(B, T, C, H, W)
        actions = torch.randn(B, T - 1, A)
        out = model(frames=frames, actions=actions)
        assert "loss" in out
        assert "pred_loss" in out
        assert "sigreg" in out
        assert "embeddings" in out
        assert "pred_embeddings" in out
        assert "target_embeddings" in out
        assert "z_std" in out
        # Loss should be a scalar
        assert out["loss"].dim() == 0

    def test_forward_all_linear(self):
        """Forward should work with all-linear encoder."""
        model = _make_slim_model(block_pattern="LLLLLL")
        model.train()
        B, T = 2, 3
        frames = torch.randn(B, T, 3, _IMG, _IMG)
        actions = torch.randn(B, T - 1, 4)
        out = model(frames=frames, actions=actions)
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_no_meta(self):
        """Forward should work with no meta tokens."""
        model = _make_slim_model(meta_tokens=0)
        model.train()
        B, T = 2, 3
        frames = torch.randn(B, T, 3, _IMG, _IMG)
        actions = torch.randn(B, T - 1, 4)
        out = model(frames=frames, actions=actions)
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# param_groups and modality
# ---------------------------------------------------------------------------

class TestHybridLeWMModelAPI:
    def test_param_groups(self):
        """param_groups should return encoder + predictor groups."""
        model = _make_slim_model()
        groups = model.param_groups()
        assert len(groups) == 2
        names = {g["name"] for g in groups}
        assert names == {"encoder", "predictor"}
        # All params should be accounted for
        group_params = sum(len(g["params"]) for g in groups)
        # Note: projector params are NOT in param_groups (same pattern as lewm)
        encoder_params = len(list(model.encoder.parameters()))
        predictor_params = len(list(model.predictor.parameters()))
        assert group_params == encoder_params + predictor_params

    def test_modality(self):
        """modality() should return 'world_model'."""
        assert HybridLeWMModel.modality() == "world_model"


# ---------------------------------------------------------------------------
# encode and predict_next
# ---------------------------------------------------------------------------

class TestHybridLeWMEncode:
    def test_encode_single_batch(self):
        """encode() should return [B, D] for a single batch of frames."""
        model = _make_slim_model()
        model.train()
        B = 4
        frames = torch.randn(B, 3, _IMG, _IMG)
        with torch.no_grad():
            z = model.encode(frames)
        assert z.shape == (B, _DIM)

    def test_encode_sequence(self):
        """encode() with [B, T, C, H, W] should return [B, T, D]."""
        model = _make_slim_model()
        model.train()
        B, T = 2, 3
        frames = torch.randn(B, T, 3, _IMG, _IMG)
        with torch.no_grad():
            z = model.encode(frames)
        assert z.shape == (B, T, _DIM)

    def test_predict_next(self):
        """predict_next should return [B, D]."""
        model = _make_slim_model()
        model.train()
        B = 4
        z = torch.randn(B, _DIM)
        action = torch.randn(B, 4)
        with torch.no_grad():
            z_next = model.predict_next(z, action)
        assert z_next.shape == (B, _DIM)


# ---------------------------------------------------------------------------
# WeightNormProjection
# ---------------------------------------------------------------------------

class TestWeightNormProjection:
    def test_apply_does_not_crash(self):
        """WeightNormProjection.apply() should run without errors."""
        model = _make_slim_model()
        hook = WeightNormProjection(model)
        hook.apply()

    def test_apply_normalizes_weights(self):
        """After apply, targeted weight rows should have norm ~sqrt(dim)."""
        model = _make_slim_model()
        hook = WeightNormProjection(model)
        hook.apply()
        # Check a random targeted weight
        if hook._targets:
            w = hook._targets[0]
            expected_norm = torch.tensor(w.shape[1], dtype=torch.float32).sqrt()
            row_norms = w.norm(dim=1)
            assert torch.allclose(row_norms, expected_norm.expand_as(row_norms), atol=1e-4)

    def test_idempotent(self):
        """Applying twice should give the same result."""
        model = _make_slim_model()
        hook = WeightNormProjection(model)
        hook.apply()
        if hook._targets:
            w1 = hook._targets[0].clone()
        hook.apply()
        if hook._targets:
            w2 = hook._targets[0]
            assert torch.allclose(w1, w2, atol=1e-5)


# ---------------------------------------------------------------------------
# Slim vs Default param count
# ---------------------------------------------------------------------------

class TestHybridLeWMSlim:
    def test_fewer_params_than_default(self):
        """A minimal hybrid should have fewer params than a larger one."""
        small = _make_slim_model()
        large = HybridLeWMModel(
            image_size=224, patch_size=14, embed_dim=192,
            encoder_depth=6, encoder_heads=3,
            block_pattern="ALALAL", meta_tokens=4,
            predictor_depth=6, predictor_heads=16,
            predictor_dim_head=64, predictor_mlp_dim=2048,
            action_dim=10, sigreg_weight=1.0,
        )
        assert _count_params(small) < _count_params(large)
