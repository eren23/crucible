"""Tests verifying scheduler factories produce correct LR curves.

Each scheduler's lr_lambda is tested against known expected values
at key points (start, mid-warmup, end-warmup, mid-decay, end).
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


def _make_mock_optimizer(base_lr: float = 0.1):
    """Create a mock optimizer with a single param group for scheduler testing."""
    opt = MagicMock()
    opt.param_groups = [{"lr": base_lr}]
    opt.defaults = {"lr": base_lr}
    return opt


class TestCosineScheduler:
    def test_cosine_no_warmup_starts_at_1(self):
        from crucible.training.schedulers import _cosine_factory
        opt = _make_mock_optimizer(0.1)
        sched = _cosine_factory(opt, total_steps=100, warmup_steps=0)
        # At step 0, lr_lambda should be ~1.0
        lr = sched.lr_lambdas[0](0)
        assert abs(lr - 1.0) < 0.01

    def test_cosine_decays_to_zero_at_end(self):
        from crucible.training.schedulers import _cosine_factory
        opt = _make_mock_optimizer(0.1)
        sched = _cosine_factory(opt, total_steps=100, warmup_steps=0, min_lr_scale=0.0)
        lr = sched.lr_lambdas[0](100)
        assert abs(lr) < 0.01

    def test_cosine_midpoint_is_half(self):
        from crucible.training.schedulers import _cosine_factory
        opt = _make_mock_optimizer(0.1)
        sched = _cosine_factory(opt, total_steps=100, warmup_steps=0, min_lr_scale=0.0)
        lr = sched.lr_lambdas[0](50)
        assert abs(lr - 0.5) < 0.01

    def test_cosine_with_warmup(self):
        from crucible.training.schedulers import _cosine_factory
        opt = _make_mock_optimizer(0.1)
        sched = _cosine_factory(opt, total_steps=1000, warmup_steps=100, min_lr_scale=0.0)
        # During warmup: linear ramp
        lr_10 = sched.lr_lambdas[0](9)  # step 9 -> (9+1)/100 = 0.1
        assert abs(lr_10 - 0.1) < 0.01
        lr_50 = sched.lr_lambdas[0](49)  # step 49 -> 50/100 = 0.5
        assert abs(lr_50 - 0.5) < 0.01
        # At warmup end: should be ~1.0
        lr_100 = sched.lr_lambdas[0](100)
        assert lr_100 > 0.99

    def test_cosine_min_lr_scale(self):
        from crucible.training.schedulers import _cosine_factory
        opt = _make_mock_optimizer(0.1)
        sched = _cosine_factory(opt, total_steps=100, warmup_steps=0, min_lr_scale=0.1)
        # At end, should decay to min_lr_scale, not 0
        lr_end = sched.lr_lambdas[0](100)
        assert abs(lr_end - 0.1) < 0.01

    def test_cosine_returns_none_when_no_steps(self):
        from crucible.training.schedulers import _cosine_factory
        opt = _make_mock_optimizer()
        result = _cosine_factory(opt, total_steps=0, warmup_steps=0)
        assert result is None


class TestConstantScheduler:
    def test_constant_no_warmup_returns_none(self):
        from crucible.training.schedulers import _constant_factory
        result = _constant_factory(_make_mock_optimizer(), total_steps=100, warmup_steps=0)
        assert result is None

    def test_constant_with_warmup_ramps(self):
        from crucible.training.schedulers import _constant_factory
        opt = _make_mock_optimizer()
        sched = _constant_factory(opt, total_steps=100, warmup_steps=10)
        # During warmup
        assert abs(sched.lr_lambdas[0](4) - 0.5) < 0.01  # (4+1)/10 = 0.5
        # After warmup: constant at 1.0
        assert abs(sched.lr_lambdas[0](50) - 1.0) < 0.01
        assert abs(sched.lr_lambdas[0](99) - 1.0) < 0.01


class TestLinearScheduler:
    def test_linear_decays_to_zero(self):
        from crucible.training.schedulers import _linear_factory
        opt = _make_mock_optimizer()
        sched = _linear_factory(opt, total_steps=100, warmup_steps=0, min_lr_scale=0.0)
        # Start
        assert abs(sched.lr_lambdas[0](0) - 1.0) < 0.01
        # Midpoint
        assert abs(sched.lr_lambdas[0](50) - 0.5) < 0.01
        # End
        assert abs(sched.lr_lambdas[0](100) - 0.0) < 0.01

    def test_linear_with_min_lr_scale(self):
        from crucible.training.schedulers import _linear_factory
        opt = _make_mock_optimizer()
        sched = _linear_factory(opt, total_steps=100, warmup_steps=0, min_lr_scale=0.2)
        lr_end = sched.lr_lambdas[0](100)
        assert abs(lr_end - 0.2) < 0.01

    def test_linear_with_warmup(self):
        from crucible.training.schedulers import _linear_factory
        opt = _make_mock_optimizer()
        sched = _linear_factory(opt, total_steps=100, warmup_steps=20, min_lr_scale=0.0)
        # During warmup: linear ramp
        lr_10 = sched.lr_lambdas[0](9)
        assert abs(lr_10 - 0.5) < 0.01  # (9+1)/20 = 0.5
        # After warmup: linear decay from 1.0 to 0.0 over remaining 80 steps
        lr_20 = sched.lr_lambdas[0](20)
        assert lr_20 > 0.99

    def test_linear_returns_none_when_no_steps(self):
        from crucible.training.schedulers import _linear_factory
        result = _linear_factory(_make_mock_optimizer(), total_steps=0, warmup_steps=0)
        assert result is None


class TestCosineRestartsScheduler:
    def test_cosine_restarts_returns_scheduler(self):
        from crucible.training.schedulers import _cosine_restarts_factory
        opt = _make_mock_optimizer()
        sched = _cosine_restarts_factory(opt, total_steps=100, warmup_steps=0, num_restarts=2)
        assert sched is not None
        # Should be a CosineAnnealingWarmRestarts instance
        import torch
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
