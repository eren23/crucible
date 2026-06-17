"""Project-name normalization shared by fleet providers and the runner.

Lives in ``core/`` (not ``fleet/``) so ``runner`` can import it without
depending on the RunPod provider — preserving the layering contract that
``fleet`` and ``runner`` stay independent of each other.
"""
from __future__ import annotations


def normalize_project_name(name: str) -> str:
    """Coerce a project name into a fleet-safe identifier.

    Replaces any character outside ``[A-Za-z0-9_-]`` with ``-``. Collapses
    runs of ``_`` so ``__`` (the project/prefix separator used for RunPod pod
    tagging) can never appear inside the normalized name itself — without
    that, project ``foo`` would falsely claim pods belonging to project
    ``foo__bar``. Empty / all-junk input returns ``""`` so callers can branch
    on the legacy un-tagged path.
    """
    if not name:
        return ""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name)
    cleaned = cleaned.strip("-_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or ""
