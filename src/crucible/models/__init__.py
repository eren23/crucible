from __future__ import annotations

from crucible.models.registry import build_model, list_families, register_model
from crucible.models.base import TiedEmbeddingLM

# Force architecture modules to self-register when the package is imported.
import crucible.models.architectures  # noqa: F401

__all__ = [
    "TiedEmbeddingLM",
    "build_model",
    "list_families",
    "register_model",
]
