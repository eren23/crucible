"""Tests for crucible.training.data_adapters."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crucible.training.data_adapters import (
    DataAdapter,
    TokenDataAdapter,
    DATA_ADAPTER_REGISTRY,
    register_data_adapter,
    build_data_adapter,
)


# ---------------------------------------------------------------------------
# DataAdapter is abstract-like
# ---------------------------------------------------------------------------


class TestDataAdapterBase:
    def test_next_batch_raises_not_implemented(self):
        adapter = DataAdapter()
        with pytest.raises(NotImplementedError):
            adapter.next_batch()

    def test_default_modality_is_generic(self):
        assert DataAdapter.modality() == "generic"


# ---------------------------------------------------------------------------
# TokenDataAdapter
# ---------------------------------------------------------------------------


class TestTokenDataAdapter:
    def test_wraps_loader(self):
        mock_loader = MagicMock()
        mock_loader.next_batch.return_value = ("input_tensor", "target_tensor")

        adapter = TokenDataAdapter(mock_loader)
        batch = adapter.next_batch(global_tokens=100, seq_len=64, grad_accum_steps=2)

        assert batch["input_ids"] == "input_tensor"
        assert batch["target_ids"] == "target_tensor"
        mock_loader.next_batch.assert_called_once_with(100, 64, 2)

    def test_modality_is_lm(self):
        assert TokenDataAdapter.modality() == "lm"

    def test_default_kwargs(self):
        mock_loader = MagicMock()
        mock_loader.next_batch.return_value = ("x", "y")

        adapter = TokenDataAdapter(mock_loader)
        batch = adapter.next_batch()

        mock_loader.next_batch.assert_called_once_with(0, 0, 1)
        assert batch["input_ids"] == "x"
        assert batch["target_ids"] == "y"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_token_registered_by_default(self):
        assert "token" in DATA_ADAPTER_REGISTRY
        assert DATA_ADAPTER_REGISTRY["token"] is TokenDataAdapter

    def test_register_custom_adapter(self):
        class MyAdapter(DataAdapter):
            def __init__(self, **kwargs):
                pass

            def next_batch(self, **kwargs):
                return {"data": "custom"}

        register_data_adapter("my_custom", MyAdapter)
        try:
            assert "my_custom" in DATA_ADAPTER_REGISTRY
            assert DATA_ADAPTER_REGISTRY["my_custom"] is MyAdapter
        finally:
            # Clean up
            DATA_ADAPTER_REGISTRY.pop("my_custom", None)

    def test_build_data_adapter_returns_instance(self):
        mock_loader = MagicMock()
        mock_loader.next_batch.return_value = ("x", "y")

        adapter = build_data_adapter("token", loader=mock_loader)
        assert isinstance(adapter, TokenDataAdapter)
        assert adapter.loader is mock_loader

    def test_build_unknown_adapter_raises(self):
        with pytest.raises(KeyError, match="Unknown data adapter"):
            build_data_adapter("nonexistent_adapter_xyz")

    def test_build_unknown_lists_available(self):
        with pytest.raises(KeyError) as exc_info:
            build_data_adapter("no_such_thing")
        assert "token" in str(exc_info.value)
