"""Group Relative Policy Optimization helpers — GRPO-style tree expansion.

GRPO (Group Relative Policy Optimization) normalizes scores within a
sampled batch and keeps the top performers by relative-advantage rather
than raw score. This decorrelates the absolute scale of any judge model
from the selection — what matters is which candidate is best *within
this group*. GIANTS (https://giants-insights.github.io/) trained their
4B insight-anticipation model with this objective.

We expose two pure helpers used by the ``tree_expand_grpo`` MCP tool:

* :func:`compute_advantages` — z-score or min-max normalization.
* :func:`select_top_k` — deterministic top-K index selection.

The orchestrator is responsible for sampling N candidates and scoring
each with its (separate) eval judge; Crucible normalizes + filters and
expands the tree with the kept children.
"""
from __future__ import annotations

import math


_VALID_NORMALIZATIONS = ("z_score", "min_max")


def compute_advantages(
    scores: list[float],
    *,
    normalization: str = "z_score",
) -> list[float]:
    """Normalize raw judge scores into group-relative advantages.

    ``z_score``: subtract mean, divide by std. Yields zero mean, unit
    variance. The standard GRPO normalization.

    ``min_max``: map to ``[0, 1]`` linearly. Useful when downstream
    code wants a ranking-style signal without negative values.

    All-equal score lists return all-zero advantages from either mode.
    """
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}. "
            f"Valid: {_VALID_NORMALIZATIONS}"
        )
    if not scores:
        return []

    if normalization == "min_max":
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span == 0:
            return [0.0] * len(scores)
        return [(s - lo) / span for s in scores]

    # z_score
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    if var == 0:
        return [0.0] * len(scores)
    std = math.sqrt(var)
    return [(s - mean) / std for s in scores]


def select_top_k(scores: list[float], *, top_k: int) -> list[int]:
    """Return indices of the ``top_k`` highest scores.

    Stable: ties are broken by ascending original index. Deterministic
    across runs. Returns an empty list when ``top_k <= 0`` or scores is
    empty; clamps when ``top_k`` exceeds ``len(scores)``.
    """
    if top_k <= 0 or not scores:
        return []
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda kv: (-kv[1], kv[0]))
    keep = indexed[: min(top_k, len(scores))]
    return [i for i, _ in keep]
