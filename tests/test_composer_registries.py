"""Tests for the composer PluginRegistry-backed registries.

These tests verify that BLOCK_TYPE_REGISTRY, AUGMENTATION_REGISTRY, and
STACK_PATTERN_REGISTRY are functional and backward-compatible with the
dict-based access pattern (BLOCK_TYPES["name"], etc.)

Requires torch — skipped when unavailable.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crucible.models.composer import (
    BLOCK_TYPE_REGISTRY,
    BLOCK_TYPES,
    AUGMENTATION_REGISTRY,
    AUGMENTATIONS,
    STACK_PATTERN_REGISTRY,
    STACK_PATTERNS,
    _ensure_block_types,
    _ensure_augmentations,
)


class TestBlockTypeRegistry:
    def test_backward_compat_dict_alias(self):
        """BLOCK_TYPES dict is the same object as the registry's internal dict."""
        assert BLOCK_TYPES is BLOCK_TYPE_REGISTRY._registry

    def test_register_custom_block_type(self):
        class FakeBlock:
            pass

        try:
            BLOCK_TYPE_REGISTRY.register("fake_block", FakeBlock, source="local")
            assert "fake_block" in BLOCK_TYPES
            assert BLOCK_TYPES["fake_block"] is FakeBlock
        finally:
            BLOCK_TYPE_REGISTRY.unregister("fake_block")


class TestAugmentationRegistry:
    def test_backward_compat_dict_alias(self):
        assert AUGMENTATIONS is AUGMENTATION_REGISTRY._registry

    def test_register_custom_augmentation(self):
        class FakeAug:
            pass

        try:
            AUGMENTATION_REGISTRY.register("fake_aug", FakeAug, source="local")
            assert "fake_aug" in AUGMENTATIONS
        finally:
            AUGMENTATION_REGISTRY.unregister("fake_aug")


class TestStackPatternRegistry:
    def test_backward_compat_dict_alias(self):
        assert STACK_PATTERNS is STACK_PATTERN_REGISTRY._registry

    def test_builtins_registered(self):
        names = STACK_PATTERN_REGISTRY.list_plugins()
        assert "sequential" in names
        assert "encoder_decoder_skip" in names
        assert "looped" in names
        assert "prefix_memory_stack" in names

    def test_register_custom_pattern(self):
        class FakePattern:
            pass

        try:
            STACK_PATTERN_REGISTRY.register("fake_pattern", FakePattern(), source="local")
            assert "fake_pattern" in STACK_PATTERNS
        finally:
            STACK_PATTERN_REGISTRY.unregister("fake_pattern")
