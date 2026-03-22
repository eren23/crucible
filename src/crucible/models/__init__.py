from __future__ import annotations

from crucible.models.registry import build_model, get_family_schema, list_families, register_model, register_schema
from crucible.models.base import TiedEmbeddingLM

# Force architecture modules to self-register when the package is imported.
import crucible.models.architectures  # noqa: F401

__all__ = [
    "TiedEmbeddingLM",
    "build_model",
    "get_family_schema",
    "list_families",
    "register_model",
    "register_schema",
]
