"""Tests for crucible.training.compression_utils."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from crucible.training.compression_utils import (
    CompressionMetrics,
    apply_weight_mask,
    iter_prunable_layers,
    make_masks_permanent,
)


def _make_model() -> nn.Sequential:
    """Simple two-layer linear model for testing."""
    return nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))


class TestSparsity:
    def test_sparsity_all_nonzero(self):
        model = _make_model()
        # Fill with non-zero values to guarantee no accidental zeros
        for p in model.parameters():
            p.data.fill_(1.0)
        result = CompressionMetrics.sparsity(model)
        assert result["sparsity/overall"] == 0.0

    def test_sparsity_with_zeros(self):
        model = _make_model()
        # Fill everything with ones, then zero out the first layer weight
        for p in model.parameters():
            p.data.fill_(1.0)
        # Zero out the first Linear weight (10*20 = 200 params)
        model[0].weight.data.zero_()
        result = CompressionMetrics.sparsity(model)
        # Total params: weight0(200) + bias0(20) + weight1(100) + bias1(5) = 325
        # Zeros: 200
        expected = 200 / 325
        assert abs(result["sparsity/overall"] - expected) < 1e-6


class TestEffectiveBits:
    def test_effective_bits_default(self):
        model = _make_model()
        bits = CompressionMetrics.effective_bits(model)
        assert bits == 32.0

    def test_effective_bits_with_assignments(self):
        model = _make_model()
        # Assign 8-bit to first layer, default 32 for the rest
        bits = CompressionMetrics.effective_bits(model, bit_assignments={"0.weight": 8})
        # weight0 = 200 params at 8 bits, bias0 = 20 at 32, weight1 = 100 at 32, bias1 = 5 at 32
        total = 200 + 20 + 100 + 5
        weighted = 200 * 8 + 20 * 32 + 100 * 32 + 5 * 32
        expected = weighted / total
        assert abs(bits - expected) < 1e-4


class TestModelSizeBytes:
    def test_model_size_bytes(self):
        model = _make_model()
        size = CompressionMetrics.model_size_bytes(model)
        # All float32 (4 bytes each)
        # weight0: 10*20=200, bias0: 20, weight1: 20*5=100, bias1: 5 → 325 params
        expected = 325 * 4
        assert size == expected


class TestNonzeroParams:
    def test_nonzero_params(self):
        model = _make_model()
        for p in model.parameters():
            p.data.fill_(1.0)
        # Zero out first layer weight
        model[0].weight.data.zero_()
        nz = CompressionMetrics.nonzero_params(model)
        # 325 total - 200 zeroed = 125
        assert nz == 125


class TestCompressionRatio:
    def test_compression_ratio(self):
        model = _make_model()
        for p in model.parameters():
            p.data.fill_(1.0)
        # Zero out first layer weight (200 of 325 params)
        model[0].weight.data.zero_()
        original_size = 325 * 4  # original bytes (all float32)
        ratio = CompressionMetrics.compression_ratio(original_size, model)
        # Non-zero: 125 params, effective bits: 32, effective bytes = 125*32/8 = 500
        expected = original_size / 500.0
        assert abs(ratio - expected) < 1e-6


class TestIterPrunableLayers:
    def test_iter_prunable_layers_includes_linear(self):
        model = _make_model()
        layers = list(iter_prunable_layers(model, exclude_patterns=()))
        assert len(layers) == 2
        for name, module in layers:
            assert isinstance(module, nn.Linear)

    def test_iter_prunable_layers_excludes_patterns(self):
        """Layers whose names match exclude patterns should be skipped."""
        # Build a model with named submodules that match exclude patterns
        model = nn.Module()
        model.tok_emb = nn.Linear(10, 20)  # should be excluded by default
        model.hidden = nn.Linear(20, 20)   # should be included
        model.lm_head = nn.Linear(20, 10)  # should be excluded by default
        layers = list(iter_prunable_layers(model))
        names = [n for n, _ in layers]
        assert "hidden" in names
        assert "tok_emb" not in names
        assert "lm_head" not in names


class TestApplyWeightMask:
    def test_apply_weight_mask(self):
        layer = nn.Linear(4, 3)
        layer.weight.data.fill_(2.0)
        mask = torch.ones_like(layer.weight)
        mask[0, :] = 0  # zero out first output row
        handle = apply_weight_mask(layer, mask)
        # Run a forward pass to trigger the hook
        x = torch.ones(1, 4)
        _ = layer(x)
        # First row of weight should now be zero
        assert (layer.weight.data[0] == 0).all()
        # Other rows should still be 2.0
        assert (layer.weight.data[1:] == 2.0).all()
        handle.remove()


class TestMakeMasksPermanent:
    def test_make_masks_permanent(self):
        model = _make_model()
        model[0].weight.data.fill_(3.0)
        mask = torch.ones_like(model[0].weight)
        mask[:5, :] = 0  # zero out first 5 output rows
        make_masks_permanent(model, {"0": mask})
        # First 5 rows should be permanently zero
        assert (model[0].weight.data[:5] == 0).all()
        # Remaining rows should still be 3.0
        assert (model[0].weight.data[5:] == 3.0).all()
