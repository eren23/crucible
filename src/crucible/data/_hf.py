"""Lazy access to the optional ``huggingface_hub`` dependency for the data pipeline."""
from __future__ import annotations

from collections.abc import Callable


def lazy_hf_hub_download() -> Callable[..., str]:
    """Return ``huggingface_hub.hf_hub_download``, imported lazily.

    Keeps ``huggingface_hub`` an optional dependency: the import happens only
    when a download is actually requested.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for dataset downloads.  "
            "Install it with: pip install huggingface-hub"
        ) from exc
    return hf_hub_download
