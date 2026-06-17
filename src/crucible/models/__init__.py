from __future__ import annotations

# Force architecture modules to self-register when the package is imported.
import crucible.models.architectures  # noqa: F401
from crucible.models.base import CrucibleModel, TiedEmbeddingLM
from crucible.models.registry import build_model, get_family_schema, list_families, register_model, register_schema

__all__ = [
    "CrucibleModel",
    "TiedEmbeddingLM",
    "build_model",
    "get_family_schema",
    "list_families",
    "register_model",
    "register_schema",
]
