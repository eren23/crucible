"""Data adapters -- uniform batch interface for all modalities.

Each adapter wraps a modality-specific data source and returns batches as
``dict[str, Tensor]`` so the generic training backend can be data-agnostic.
"""
from __future__ import annotations

from typing import Any


class DataAdapter:
    """Base class for data adapters."""

    def next_batch(self, **kwargs: Any) -> dict[str, Any]:
        """Return the next training batch as a dict of tensors."""
        raise NotImplementedError

    @classmethod
    def modality(cls) -> str:
        """Modality tag matching the model it feeds."""
        return "generic"


class TokenDataAdapter(DataAdapter):
    """Wraps the existing :class:`DistributedTokenLoader`.

    Translates ``(x, y)`` tuples into ``{'input_ids': x, 'target_ids': y}``.
    """

    def __init__(self, loader: Any):
        self.loader = loader

    def next_batch(
        self,
        global_tokens: int = 0,
        seq_len: int = 0,
        grad_accum_steps: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        x, y = self.loader.next_batch(global_tokens, seq_len, grad_accum_steps)
        return {"input_ids": x, "target_ids": y}

    @classmethod
    def modality(cls) -> str:
        return "lm"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATA_ADAPTER_REGISTRY: dict[str, type[DataAdapter]] = {}


def register_data_adapter(name: str, cls: type[DataAdapter]) -> None:
    """Register a data adapter class under *name*."""
    DATA_ADAPTER_REGISTRY[name] = cls


def build_data_adapter(name: str, **kwargs: Any) -> DataAdapter:
    """Instantiate a registered data adapter by name."""
    cls = DATA_ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown data adapter '{name}'. Available: {sorted(DATA_ADAPTER_REGISTRY)}"
        )
    return cls(**kwargs)


# Register built-ins
register_data_adapter("token", TokenDataAdapter)
