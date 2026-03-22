"""Tests for crucible.models.components — individual component unit tests."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crucible.models.components.norm import RMSNorm
from crucible.models.components.linear import CastedLinear
from crucible.models.components.mlp import MLP, ACTIVATIONS
from crucible.models.components.moe import MoELayer
from crucible.models.components.conv import DepthwiseConv1D
from crucible.models.components.gate import SmearGate


DIM = 64
BATCH = 2
SEQ = 16


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm()
        x = torch.randn(BATCH, SEQ, DIM)
        out = norm(x)
        assert out.shape == (BATCH, SEQ, DIM)

    def test_normalizes(self):
        norm = RMSNorm()
        x = torch.randn(BATCH, SEQ, DIM) * 10
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestCastedLinear:
    def test_output_shape(self):
        linear = CastedLinear(DIM, DIM * 2, bias=False)
        x = torch.randn(BATCH, SEQ, DIM)
        out = linear(x)
        assert out.shape == (BATCH, SEQ, DIM * 2)

    def test_with_bias(self):
        linear = CastedLinear(DIM, DIM, bias=True)
        x = torch.randn(BATCH, SEQ, DIM)
        out = linear(x)
        assert out.shape == (BATCH, SEQ, DIM)


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(DIM, DIM * 2)
        x = torch.randn(BATCH, SEQ, DIM)
        out = mlp(x)
        assert out.shape == (BATCH, SEQ, DIM)

    @pytest.mark.parametrize("activation", list(ACTIVATIONS.keys()))
    def test_all_activations(self, activation: str):
        mlp = MLP(DIM, DIM * 2, activation=activation)
        x = torch.randn(BATCH, SEQ, DIM)
        out = mlp(x)
        assert out.shape == (BATCH, SEQ, DIM)
        assert torch.isfinite(out).all()


class TestMoELayer:
    def test_output_shape(self):
        moe = MoELayer(DIM, num_experts=4, top_k=2, mlp_mult=2)
        x = torch.randn(BATCH, SEQ, DIM)
        out = moe(x)
        assert out.shape == (BATCH, SEQ, DIM)

    def test_aux_loss_during_training(self):
        moe = MoELayer(DIM, num_experts=4, top_k=2)
        moe.train()
        x = torch.randn(BATCH, SEQ, DIM)
        _ = moe(x)
        assert moe.aux_loss.dim() == 0
        assert moe.aux_loss.item() >= 0

    def test_no_aux_loss_during_inference(self):
        moe = MoELayer(DIM, num_experts=4, top_k=2)
        moe.eval()
        x = torch.randn(BATCH, SEQ, DIM)
        _ = moe(x)
        assert moe.aux_loss.item() == 0.0


class TestDepthwiseConv1D:
    def test_output_shape(self):
        conv = DepthwiseConv1D(DIM, kernel=3)
        x = torch.randn(BATCH, SEQ, DIM)
        out = conv(x)
        assert out.shape == (BATCH, SEQ, DIM)

    def test_causal(self):
        """Output at position t should not depend on future positions."""
        conv = DepthwiseConv1D(DIM, kernel=3)
        x = torch.randn(1, 8, DIM)
        out_full = conv(x)
        out_prefix = conv(x[:, :4, :])
        assert torch.allclose(out_full[:, :4, :], out_prefix, atol=1e-5)


class TestSmearGate:
    def test_output_shape(self):
        gate = SmearGate(DIM)
        x = torch.randn(BATCH, SEQ, DIM)
        out = gate(x)
        assert out.shape == (BATCH, SEQ, DIM)
