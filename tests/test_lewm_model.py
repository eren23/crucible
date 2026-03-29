"""Tests for the LE-WM architecture plugin."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_lewm_path = Path(__file__).parent.parent / ".crucible/plugins/architectures/lewm.py"
if not _lewm_path.exists():
    pytest.skip(f"LE-WM plugin not found at {_lewm_path}", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("lewm", str(_lewm_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LeWMModel = _mod.LeWMModel


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_default_model() -> "LeWMModel":
    """Build a LE-WM with default config (ViT-Tiny scale)."""
    return LeWMModel(
        image_size=224,
        patch_size=14,
        in_channels=3,
        embed_dim=192,
        encoder_depth=6,
        encoder_heads=3,
        predictor_depth=6,
        predictor_heads=16,
        predictor_dim_head=64,
        predictor_mlp_dim=2048,
        action_dim=10,
        sigreg_weight=1.0,
        sigreg_projections=128,
        dropout=0.1,
    )


def _make_slim_model() -> "LeWMModel":
    """Build a smaller LE-WM for faster tests."""
    return LeWMModel(
        image_size=56,
        patch_size=14,
        in_channels=3,
        embed_dim=64,
        encoder_depth=2,
        encoder_heads=2,
        predictor_depth=2,
        predictor_heads=4,
        predictor_dim_head=16,
        predictor_mlp_dim=128,
        action_dim=4,
        sigreg_weight=1.0,
        sigreg_projections=32,
        dropout=0.0,
    )


class TestLeWMParamCount:
    def test_lewm_param_count(self):
        """Default config should produce a model in the ~15M param range."""
        model = _make_default_model()
        n = _count_params(model)
        # Expect roughly 10-25M params for the default ViT-Tiny + DiT predictor
        assert 5_000_000 < n < 30_000_000, f"Expected ~15M params, got {n:,}"


class TestLeWMForward:
    def test_lewm_forward_output_keys(self):
        """Forward should return loss, pred_loss, sigreg, embeddings, etc."""
        model = _make_slim_model()
        model.train()  # BatchNorm needs train mode for batch stats
        B, T = 2, 3
        C, H, W = 3, 56, 56
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


class TestLeWMEncode:
    def test_lewm_encode(self):
        """encode() should return correct shape [B, D] for a single batch of frames."""
        model = _make_slim_model()
        model.train()
        B, C, H, W = 4, 3, 56, 56
        frames = torch.randn(B, C, H, W)
        with torch.no_grad():
            z = model.encode(frames)
        assert z.shape == (B, 64)  # embed_dim=64

    def test_lewm_encode_sequence(self):
        """encode() with [B, T, C, H, W] should return [B, T, D]."""
        model = _make_slim_model()
        model.train()
        B, T, C, H, W = 2, 3, 3, 56, 56
        frames = torch.randn(B, T, C, H, W)
        with torch.no_grad():
            z = model.encode(frames)
        assert z.shape == (B, T, 64)


class TestLeWMPredictNext:
    def test_lewm_predict_next(self):
        """predict_next should return correct shape [B, D]."""
        model = _make_slim_model()
        model.train()
        B = 4
        z = torch.randn(B, 64)      # embed_dim=64
        action = torch.randn(B, 4)   # action_dim=4
        with torch.no_grad():
            z_next = model.predict_next(z, action)
        assert z_next.shape == (B, 64)


class TestLeWMSlim:
    def test_lewm_slim_fewer_params(self):
        """Reduced dim/depth should give fewer params than default."""
        default_model = _make_default_model()
        slim_model = _make_slim_model()
        default_params = _count_params(default_model)
        slim_params = _count_params(slim_model)
        assert slim_params < default_params
